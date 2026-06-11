"""
reqgraph.tiling
===============

The lossless engine. Turns a list of *claims* (typed character spans proposed by
any extractor) into a RequirementGraph.

Claim = (start: int, end: int, role: Role, attrs: dict)

Guarantee: the resulting graph's terminal nodes *tile* the input text -- every
character is owned by exactly one terminal node (semantic element or GLUE), so
``graph.generate()`` reproduces the input byte-for-byte regardless of how good
the extractor was. The semantic tree (ROOT -> ACTION -> PROCESS/OBJECT, logical
AND/OR operators, conditions, constraints) is layered on top.
"""

from __future__ import annotations

import logging
from typing import Optional

from .core import Node, RequirementGraph, Rel, Role

logger = logging.getLogger(__name__)

Claim = tuple[int, int, Role, dict]


def _sanitise(claims: list[Claim], text_len: int) -> list[Claim]:
    """Clamp spans to [0, len(text)] and drop empties/junk.

    Extractors are untrusted plugins: a buggy backend must never be able to
    crash the tiler or break the lossless invariant."""
    out: list[Claim] = []
    for c in claims:
        try:
            s, e, role, attrs = c
            s, e = int(s), int(e)
        except (TypeError, ValueError):
            logger.warning("dropping malformed claim %r", c)
            continue
        s = max(0, min(s, text_len))
        e = max(0, min(e, text_len))
        if e > s and isinstance(role, Role):
            out.append((s, e, role, dict(attrs or {})))
        else:
            logger.debug("dropping empty/invalid claim %r", c)
    return out


def _dedup_nonoverlap(claims: list[Claim]) -> list[Claim]:
    """Keep earliest-starting claims; drop any that overlap an accepted one.

    Longer spans win ties at the same start (more specific first)."""
    claims = sorted(claims, key=lambda c: (c[0], -(c[1] - c[0])))
    accepted: list[Claim] = []
    last_end = -1
    for (s, e, role, attrs) in claims:
        if e > s and s >= last_end:
            accepted.append((s, e, role, attrs))
            last_end = e
    return accepted


def _infer_operator(text: str, gap_start: int, gap_end: int) -> str:
    """Pick AND/OR from the connective text between two actions."""
    gap = text[gap_start:gap_end].lower()
    if " or " in f" {gap.strip()} ":
        return "OR"
    return "AND"


def tile_to_graph(text: str, claims: list[Claim], template_name: str = "",
                  metadata: Optional[dict] = None) -> RequirementGraph:
    g = RequirementGraph(template_name, metadata)
    accepted = _dedup_nonoverlap(_sanitise(claims, len(text)))

    # --- tile the text with terminals (semantic spans + GLUE gaps) ----------
    terminals: list[Node] = []
    span_node: dict[tuple, Node] = {}
    pos = 0
    for (s, e, role, attrs) in accepted:
        if s > pos:
            terminals.append(g.add_node(Role.GLUE, text[pos:s]))
        node = g.add_node(role, text[s:e], **attrs)
        span_node[(s, e)] = node
        terminals.append(node)
        pos = e
    if pos < len(text):
        terminals.append(g.add_node(Role.GLUE, text[pos:]))

    g.leaf_order = [n.id for n in terminals]
    for a, b in zip(terminals, terminals[1:]):
        g.add_edge(a, b, Rel.NEXT)

    # --- semantic tree on top of the terminals ------------------------------
    root = g.add_node(Role.ROOT, **(metadata or {}))
    g.root_id = root.id

    def nodes_of(role):
        return [(s, e, span_node[(s, e)]) for (s, e, r, _a) in accepted if r is role]

    for _s, _e, c in nodes_of(Role.CONDITION):
        g.add_edge(root, c, Rel.HAS_CONDITION)
    for _s, _e, sub in nodes_of(Role.SUBJECT):
        g.add_edge(root, sub, Rel.HAS_SUBJECT)
    for _s, _e, m in nodes_of(Role.MODALITY):
        g.add_edge(root, m, Rel.HAS_MODALITY)
    actors = nodes_of(Role.ACTOR)
    constraints = nodes_of(Role.CONSTRAINT)
    details = nodes_of(Role.DETAILS)

    # Build ACTION groups by walking PROCESS/OBJECT in text order: each PROCESS
    # starts a new action; following OBJECT(s) attach to it.
    seq = [(s, e, r, a) for (s, e, r, a) in accepted if r in (Role.PROCESS, Role.OBJECT)]
    actions = []  # (action_node, start_extent, end_extent)
    current = None
    for (s, e, r, a) in seq:
        if r is Role.PROCESS:
            current = g.add_node(Role.ACTION,
                                 function_type=a.get("function_type", "autonomous activity"))
            g.add_edge(current, span_node[(s, e)], Rel.HAS_PROCESS)
            actions.append([current, s, e])
        else:  # OBJECT
            if current is None:
                current = g.add_node(Role.ACTION)
                actions.append([current, s, e])
            g.add_edge(current, span_node[(s, e)], Rel.ACTS_ON)
            actions[-1][2] = e

    action_nodes = [a[0] for a in actions]
    head = None
    if len(action_nodes) == 1:
        g.add_edge(root, action_nodes[0], Rel.HAS_ACTION)
        head = action_nodes[0]
    elif len(action_nodes) > 1:
        op_word = _infer_operator(text, actions[0][2], actions[1][1])
        op = g.add_node(Role.OPERATOR, operator=op_word)
        g.add_edge(root, op, Rel.HAS_ACTION)
        for a in action_nodes:
            g.add_edge(op, a, Rel.OPERAND)
        head = action_nodes[0]

    anchor = head or root
    for _s, _e, ac in actors:
        g.add_edge(anchor, ac, Rel.HAS_ACTOR)
    for _s, _e, dn in details:
        g.add_edge(anchor, dn, Rel.HAS_DETAILS)
    for _s, _e, cn in constraints:
        g.add_edge(anchor, cn, Rel.HAS_CONSTRAINT)

    return g
