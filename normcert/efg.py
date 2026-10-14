"""Self-contained vectorized two-player zero-sum poker EFGs (Kuhn, Leduc)
with an extra-gradient IR-PCFR+ solver (one IR-PRM+ instance per infoset).

Public-tree representation: nodes are public histories; values are
[n_hands0 x n_hands1] matrices (P0 view, conditional on hands, including
downstream chance weights but not deal probabilities or player reach).
Infoset = (public player node, private hand); each player node holds a
_BatchIRPRM over hands.

Kuhn: 3 cards, ante 1, bet 1, max 1 raise.
Leduc: 3 ranks x 2 suits, ante 1, round-1 raise 2, round-2 raise 4,
max 2 raises per round, one public card between rounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ireg_prm import _BatchIRPRM


# ---------------------------------------------------------------------------
# public tree
# ---------------------------------------------------------------------------

class PubNode:
    __slots__ = ("kind", "player", "actions", "children", "payoff",
                 "chance_w", "node_id", "label")

    def __init__(self, kind: str, label: str = "") -> None:
        self.kind = kind              # 'player' | 'chance' | 'terminal'
        self.player: int = -1
        self.actions: List[str] = []
        self.children: List["PubNode"] = []
        self.payoff: Optional[np.ndarray] = None   # terminal [n0, n1]
        self.chance_w: List[np.ndarray] = []       # per-branch cond. weight
        self.node_id: int = -1
        self.label = label


@dataclass
class EFG:
    name: str
    root: PubNode
    n_hands: int
    deal_w: np.ndarray                  # [n0, n1] joint deal probability
    player_nodes: List[PubNode] = field(default_factory=list)

    def index_nodes(self) -> None:
        self.player_nodes = []
        stack = [self.root]
        while stack:
            nd = stack.pop()
            if nd.kind == "player":
                nd.node_id = len(self.player_nodes)
                self.player_nodes.append(nd)
            stack.extend(nd.children)


# ---------------------------------------------------------------------------
# game builders
# ---------------------------------------------------------------------------

def _build_betting(n_hands: int, to_act: int, checked: bool,
                   outstanding: float, n_raises: int, max_raises: int,
                   raise_size: float, contribs: Tuple[float, float],
                   cont_fn: Callable[[Tuple[float, float]], PubNode],
                   fold_fn: Callable[[int, Tuple[float, float]], PubNode],
                   label: str) -> PubNode:
    """Generic single-round betting tree.

    outstanding > 0: to_act faces a bet; else may check or raise.
    cont_fn(contribs): round finished with equal contributions.
    fold_fn(folder, contribs): fold terminal.
    """
    nd = PubNode("player", label)
    nd.player = to_act
    opp = 1 - to_act
    if outstanding > 0:
        nd.actions.append("f")
        nd.children.append(fold_fn(to_act, contribs))
        # call
        c2 = list(contribs)
        c2[to_act] += outstanding
        nd.actions.append("c")
        nd.children.append(cont_fn(tuple(c2)))
        if n_raises < max_raises:
            c3 = list(contribs)
            c3[to_act] += outstanding + raise_size
            nd.actions.append("r")
            nd.children.append(_build_betting(
                n_hands, opp, False, raise_size, n_raises + 1, max_raises,
                raise_size, tuple(c3), cont_fn, fold_fn, label + "r"))
    else:
        # check (if both checked -> continue)
        if checked:
            nd.actions.append("k")
            nd.children.append(cont_fn(contribs))
        else:
            nd.actions.append("k")
            nd.children.append(_build_betting(
                n_hands, opp, True, 0.0, n_raises, max_raises, raise_size,
                contribs, cont_fn, fold_fn, label + "k"))
        if n_raises < max_raises:
            c3 = list(contribs)
            c3[to_act] += raise_size
            nd.actions.append("r")
            nd.children.append(_build_betting(
                n_hands, opp, False, raise_size, n_raises + 1, max_raises,
                raise_size, tuple(c3), cont_fn, fold_fn, label + "r"))
    return nd


def _fold_terminal(folder: int, contribs: Tuple[float, float],
                   n: int) -> PubNode:
    nd = PubNode("terminal")
    lost = contribs[folder]
    nd.payoff = np.full((n, n), lost if folder == 1 else -lost)
    return nd


def build_kuhn() -> EFG:
    n = 3
    deal = (np.ones((n, n)) - np.eye(n)) / 6.0

    def showdown(contribs: Tuple[float, float]) -> PubNode:
        nd = PubNode("terminal")
        stake = min(contribs)  # equal at showdown
        h0 = np.arange(n)[:, None]
        h1 = np.arange(n)[None, :]
        nd.payoff = np.where(h0 > h1, stake, -stake) * 1.0
        np.fill_diagonal(nd.payoff, 0.0)
        return nd

    root = _build_betting(n, 0, False, 0.0, 0, 1, 1.0, (1.0, 1.0),
                          showdown, lambda f, c: _fold_terminal(f, c, n), "")
    g = EFG("kuhn", root, n, deal)
    g.index_nodes()
    return g


def build_leduc(n_ranks: int = 3, n_suits: int = 2) -> EFG:
    n = n_ranks * n_suits
    rank = np.arange(n) // n_suits
    deal = (np.ones((n, n)) - np.eye(n)) / (n * (n - 1))

    def showdown(board: int, contribs: Tuple[float, float]) -> PubNode:
        nd = PubNode("terminal")
        stake = min(contribs)
        r0 = rank[:, None]
        r1 = rank[None, :]
        rb = rank[board]
        pair0 = (r0 == rb)
        pair1 = (r1 == rb)
        win = np.where(pair0 & ~pair1, 1.0,
              np.where(pair1 & ~pair0, -1.0,
              np.where(r0 > r1, 1.0, np.where(r1 > r0, -1.0, 0.0))))
        nd.payoff = win * stake
        return nd

    def round2(board: int, contribs: Tuple[float, float]) -> PubNode:
        return _build_betting(
            n, 0, False, 0.0, 0, 2, 4.0, contribs,
            lambda c: showdown(board, c),
            lambda f, c: _fold_terminal(f, c, n), f"b{board}|")

    def chance(contribs: Tuple[float, float]) -> PubNode:
        nd = PubNode("chance")
        for b in range(n):
            w = np.ones((n, n)) / (n - 2)
            w[b, :] = 0.0
            w[:, b] = 0.0
            nd.chance_w.append(w)
            nd.children.append(round2(b, contribs))
        return nd

    root = _build_betting(n, 0, False, 0.0, 0, 2, 2.0, (1.0, 1.0),
                          chance, lambda f, c: _fold_terminal(f, c, n), "")
    g = EFG("leduc", root, n, deal)
    g.index_nodes()
    return g


# ---------------------------------------------------------------------------
# vectorized profile pass: values + counterfactual utilities + reaches
# ---------------------------------------------------------------------------

def profile_pass(g: EFG, strat: List[np.ndarray],
                 ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """One tree pass under `strat` (list over player_nodes: [n_hands, n_act]).

    Returns (root value matrix V[h0,h1], cf-utility list u[node_id]
    ([n_hands, n_act], own-player view), my-reach list [n_hands]).
    """
    n = g.n_hands
    cf_u: List[Optional[np.ndarray]] = [None] * len(g.player_nodes)
    my_reach: List[Optional[np.ndarray]] = [None] * len(g.player_nodes)

    def rec(nd: PubNode, r0: np.ndarray, r1: np.ndarray, W: np.ndarray,
            ) -> np.ndarray:
        if nd.kind == "terminal":
            return nd.payoff
        if nd.kind == "chance":
            V = np.zeros((n, n))
            for w, ch in zip(nd.chance_w, nd.children):
                V += w * rec(ch, r0, r1, W * w)
            return V
        S = strat[nd.node_id]
        p = nd.player
        Vas = []
        for a, ch in enumerate(nd.children):
            if p == 0:
                Vas.append(rec(ch, r0 * S[:, a], r1, W))
            else:
                Vas.append(rec(ch, r0, r1 * S[:, a], W))
        if p == 0:
            u = np.stack([(Va * W) @ r1 for Va in Vas], axis=1)
            my_reach[nd.node_id] = r0
        else:
            u = np.stack([-(r0 @ (Va * W)) for Va in Vas], axis=1)
            my_reach[nd.node_id] = r1
        cf_u[nd.node_id] = u
        V = np.zeros((n, n))
        for a, Va in enumerate(Vas):
            if p == 0:
                V += S[:, a][:, None] * Va
            else:
                V += S[:, a][None, :] * Va
        return V

    ones = np.ones(n)
    V = rec(g.root, ones, ones, g.deal_w)
    return V, cf_u, my_reach  # type: ignore[return-value]


def best_response_value(g: EFG, strat: List[np.ndarray], player: int,
                        ) -> float:
    """Value of `player`'s best response against strat[opponent].

    Returns sum over own hands of BR value (own view)."""
    n = g.n_hands

    def rec(nd: PubNode, w: np.ndarray) -> np.ndarray:
        # w[h0,h1]: joint weight = deal * chance-so-far * opp reach
        if nd.kind == "terminal":
            P = nd.payoff if player == 0 else -nd.payoff
            return (P * w).sum(axis=1 - player)
        if nd.kind == "chance":
            bv = np.zeros(n)
            for cw, ch in zip(nd.chance_w, nd.children):
                bv += rec(ch, w * cw)
            return bv
        S = strat[nd.node_id]
        if nd.player == player:
            vals = [rec(ch, w) for ch in nd.children]
            return np.max(np.stack(vals, axis=0), axis=0)
        bv = np.zeros(n)
        for a, ch in enumerate(nd.children):
            wa = w * (S[:, a][:, None] if nd.player == 0
                      else S[:, a][None, :])
            bv += rec(ch, wa)
        return bv

    return float(rec(g.root, g.deal_w).sum())


def nash_gap(g: EFG, strat: List[np.ndarray]) -> float:
    """Duality gap: BR0(strat1) + BR1(strat0) - 0 (zero-sum)."""
    return best_response_value(g, strat, 0) + best_response_value(g, strat, 1)


# ---------------------------------------------------------------------------
# extra-gradient IR-PCFR+ solver
# ---------------------------------------------------------------------------

@dataclass
class EFGTrace:
    ts: np.ndarray
    norm1: np.ndarray            # global l2 over player-0 infosets
    norm2: np.ndarray
    gap_last: np.ndarray
    gap_best: np.ndarray
    gap_avg: np.ndarray
    infoset_norms: np.ndarray    # (K, total_infosets) per-infoset norms
    infoset_meta: List[Tuple[int, int, int]]  # (node_id, player, hand)
    infoset_reach: np.ndarray    # avg my-reach per infoset (time-avg)
    cum_g2: np.ndarray           # cumulative sum ||g_i||^2 over infosets


def run_ireg_pcfr(g: EFG, T: int, record_at: np.ndarray) -> EFGTrace:
    minimizers = [_BatchIRPRM(g.n_hands, len(nd.actions))
                  for nd in g.player_nodes]
    avg = [np.zeros((g.n_hands, len(nd.actions))) for nd in g.player_nodes]
    meta: List[Tuple[int, int, int]] = []
    for nd in g.player_nodes:
        for h in range(g.n_hands):
            meta.append((nd.node_id, nd.player, h))
    rec_set = set(int(v) for v in record_at)
    ts: List[int] = []
    rec: Dict[str, List] = {k: [] for k in ("n1", "n2", "gl", "gb", "ga",
                                            "inorm", "cg2")}
    best = np.inf
    cum_g2 = 0.0
    reach_acc = [np.zeros(g.n_hands) for _ in g.player_nodes]
    for t in range(1, T + 1):
        stilde = [mz.xtilde for mz in minimizers]
        _, m_u, _ = profile_pass(g, stilde)
        x = [mz.next_strategy(m_u[i]) for i, mz in enumerate(minimizers)]
        _, u, reach = profile_pass(g, x)
        for i, mz in enumerate(minimizers):
            cum_g2 += mz.observe_utility(u[i])
            avg[i] += reach[i][:, None] * x[i]
            reach_acc[i] += reach[i]
        if t in rec_set:
            gap = nash_gap(g, x)
            best = min(best, gap)
            avg_prof = []
            for i, nd in enumerate(g.player_nodes):
                s = avg[i].sum(axis=1, keepdims=True)
                unif = np.full_like(avg[i], 1.0 / len(nd.actions))
                avg_prof.append(np.where(s > 0, avg[i] / np.maximum(s, 1e-300),
                                         unif))
            gap_a = nash_gap(g, avg_prof)
            n1sq = n2sq = 0.0
            inorms = np.empty(len(meta))
            k = 0
            for i, nd in enumerate(g.player_nodes):
                nrm = np.linalg.norm(minimizers[i].rtilde, axis=1)
                inorms[k:k + g.n_hands] = nrm
                k += g.n_hands
                if nd.player == 0:
                    n1sq += float((nrm ** 2).sum())
                else:
                    n2sq += float((nrm ** 2).sum())
            ts.append(t)
            rec["n1"].append(np.sqrt(n1sq))
            rec["n2"].append(np.sqrt(n2sq))
            rec["gl"].append(gap)
            rec["gb"].append(best)
            rec["ga"].append(gap_a)
            rec["inorm"].append(inorms)
            rec["cg2"].append(cum_g2)
    reach_avg = np.concatenate([ra / T for ra in reach_acc])
    return EFGTrace(np.array(ts), np.array(rec["n1"]), np.array(rec["n2"]),
                    np.array(rec["gl"]), np.array(rec["gb"]),
                    np.array(rec["ga"]), np.stack(rec["inorm"]), meta,
                    reach_avg, np.array(rec["cg2"]))
