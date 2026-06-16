"""
reqgraph.corpus
================

Cross-requirement analysis over a *requirement set*: parse every requirement
in a batch into its own ``RequirementGraph`` (compound items are split first,
same as a single ``parse``), then find SUBJECT/OBJECT elements that recur --
near-verbatim or paraphrased -- across *different* requirements and connect
those requirements into a "connections graph" for visualization and export
(Mermaid / DOT / GraphML / Turtle / Cypher).

This surfaces hidden coupling in a requirement set: e.g. two requirements
that both govern "the fuel pump" or both act on "the flight plan" even though
nothing else ties them together -- useful for traceability review and
spotting duplicated/contradictory responsibility for the same component.

Similarity is pluggable:

* ``"lexical"`` (default) -- token-overlap (Jaccard) blended with a
  character sequence ratio. Zero dependencies, deterministic.
* ``"embedding"`` -- BERT cosine similarity via
  :class:`reqgraph.nlp.RequirementAnalyzer` (requires torch + transformers).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .core import Role, _dot_label, _mermaid_label, _ttl_literal, _xml_escape
from .parser import RequirementParser
from .templates import RUPP_TEMPLATE, Template

_WORD_RE = re.compile(r"[a-z0-9]+")
_ID_RE = re.compile(r"[^A-Za-z0-9_]")
# determiners carry no identity ("the system" vs "the engine" must not score
# high just because both happen to start with "the") -- stripped before
# comparison only; the original element text is kept for display.
_STOPWORDS = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "its", "their",
    "his", "her", "our", "your", "each", "every", "all", "any",
})


def _normalise_items(items):
    """Accept ['text', ...] or [(id, text), ...]; yield (id, text)."""
    for it in items:
        if isinstance(it, (tuple, list)):
            yield (it[0], it[1])
        else:
            yield (None, it)


def _normalise_text(s: str) -> str:
    return " ".join(w for w in _WORD_RE.findall(s.lower()) if w not in _STOPWORDS)


def _lexical_similarity(a: str, b: str) -> float:
    """Token-set Jaccard blended with a sequence ratio (covers short spans
    where Jaccard alone is noisy, e.g. single-word subjects/objects)."""
    na, nb = _normalise_text(a), _normalise_text(b)
    if not na or not nb:
        return 0.0
    ta, tb = set(na.split()), set(nb.split())
    jaccard = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    return max(jaccard, ratio)


def _safe_local_id(rid: str) -> str:
    safe = _ID_RE.sub("_", rid) or "req"
    return safe if not safe[0].isdigit() else f"r_{safe}"


@dataclass(frozen=True)
class ElementRef:
    """One SUBJECT/OBJECT (etc.) element, tagged with its source requirement."""
    req_id: str
    node_id: str
    role: Role
    text: str


@dataclass(frozen=True)
class Connection:
    """A similarity edge between two elements from different requirements."""
    a: ElementRef
    b: ElementRef
    role: Role
    score: float


class RequirementSetGraph:
    """Per-requirement graphs plus the cross-requirement connections between them."""

    def __init__(self, req_ids: list, texts: dict, graphs: dict,
                connections: list):
        self.req_ids = req_ids
        self.texts = texts
        self.graphs = graphs
        self.connections = connections

    def _index(self) -> dict:
        return {rid: i for i, rid in enumerate(self.req_ids)}

    def connections_for(self, req_id: str) -> list:
        return [c for c in self.connections
                if c.a.req_id == req_id or c.b.req_id == req_id]

    def _dedup_pairs(self):
        """One representative (highest-score) connection per (req_a, req_b, role)."""
        seen = {}
        for c in self.connections:
            key = (c.a.req_id, c.b.req_id, c.role.value)
            if key not in seen or c.score > seen[key].score:
                seen[key] = c
        return list(seen.values())

    def to_dict(self) -> dict:
        return {
            "requirements": [{"id": rid, "text": self.texts[rid]} for rid in self.req_ids],
            "connections": [
                {"req_a": c.a.req_id, "req_b": c.b.req_id, "role": c.role.value,
                 "score": round(c.score, 4), "text_a": c.a.text, "text_b": c.b.text}
                for c in self.connections
            ],
        }

    def to_mermaid(self) -> str:
        idx = self._index()
        lines = ["flowchart LR"]
        for rid in self.req_ids:
            label = self.texts[rid].strip()
            if len(label) > 50:
                label = label[:47] + "..."
            lines.append(f'    r{idx[rid]}["{_mermaid_label(rid)}: {_mermaid_label(label)}"]')
        for c in self._dedup_pairs():
            edge_label = f"{c.role.value} ~{c.score:.2f}: {c.a.text}"
            lines.append(f'    r{idx[c.a.req_id]} -->|"{_mermaid_label(edge_label)}"| '
                         f'r{idx[c.b.req_id]}')
        return "\n".join(lines)

    def to_dot(self) -> str:
        idx = self._index()
        lines = ["digraph RequirementSet {", "  rankdir=LR;",
                 '  node [fontname="Helvetica", shape=box];']
        for rid in self.req_ids:
            label = self.texts[rid].strip()
            if len(label) > 50:
                label = label[:47] + "..."
            lines.append(f'  "r{idx[rid]}" [label="{_dot_label(rid + chr(10) + label)}"];')
        for c in self._dedup_pairs():
            edge_label = f"{c.role.value} {c.score:.2f}"
            lines.append(f'  "r{idx[c.a.req_id]}" -> "r{idx[c.b.req_id]}" '
                         f'[label="{_dot_label(edge_label)}"];')
        lines.append("}")
        return "\n".join(lines)

    def to_graphml(self) -> str:
        ns = "http://graphml.graphdrawing.org/xmlns"
        lines = ['<?xml version="1.0" encoding="UTF-8"?>',
                 f'<graphml xmlns="{ns}">',
                 '  <key id="text" for="node" attr.name="text" attr.type="string"/>',
                 '  <key id="role" for="edge" attr.name="role" attr.type="string"/>',
                 '  <key id="score" for="edge" attr.name="score" attr.type="double"/>',
                 '  <key id="match" for="edge" attr.name="match" attr.type="string"/>',
                 '  <graph id="requirement-set" edgedefault="directed">']
        for rid in self.req_ids:
            lines.append(f'    <node id="{_xml_escape(rid)}">')
            lines.append(f'      <data key="text">{_xml_escape(self.texts[rid].strip())}</data>')
            lines.append('    </node>')
        for i, c in enumerate(self._dedup_pairs()):
            lines.append(f'    <edge id="c{i}" source="{_xml_escape(c.a.req_id)}" '
                         f'target="{_xml_escape(c.b.req_id)}">')
            lines.append(f'      <data key="role">{_xml_escape(c.role.value)}</data>')
            lines.append(f'      <data key="score">{c.score:.4f}</data>')
            lines.append(f'      <data key="match">{_xml_escape(c.a.text)}</data>')
            lines.append('    </edge>')
        lines.append('  </graph>')
        lines.append('</graphml>')
        return "\n".join(lines)

    def to_turtle(self, base: str = "http://reqgraph.org/ns#") -> str:
        prefix = "rg"
        lines = [f"@prefix {prefix}: <{base}> .",
                 "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .", ""]
        for rid in self.req_ids:
            lit = f'"{_ttl_literal(self.texts[rid].strip())}"'
            lines.append(f"{prefix}:{_safe_local_id(rid)} a {prefix}:Requirement ;\n"
                         f"    rdfs:label {lit} .")
        lines.append("")
        for c in self._dedup_pairs():
            pred = f"similar{c.role.value.capitalize()}"
            lines.append(f"{prefix}:{_safe_local_id(c.a.req_id)} {prefix}:{pred} "
                         f"{prefix}:{_safe_local_id(c.b.req_id)} .")
        return "\n".join(lines)

    def to_cypher(self) -> str:
        stmts = []
        for rid in self.req_ids:
            props = {"id": rid, "text": self.texts[rid].strip()}
            prop_str = ", ".join(f"{k}: {json.dumps(v)}" for k, v in props.items())
            stmts.append(f"CREATE (:Requirement {{{prop_str}}})")
        for c in self._dedup_pairs():
            rel = f"SIMILAR_{c.role.value}"
            stmts.append(
                f"MATCH (a:Requirement {{id:{json.dumps(c.a.req_id)}}}),"
                f"(b:Requirement {{id:{json.dumps(c.b.req_id)}}}) "
                f"CREATE (a)-[:{rel} {{score: {c.score:.4f}}}]->(b)")
        return "\n".join(stmts)


def _score_fn(elements: list, similarity: str, embedding_model: str):
    if similarity == "embedding":
        from .nlp import RequirementAnalyzer
        az = RequirementAnalyzer(model_name=embedding_model)
        emb = az.embed([e.text for e in elements]) if elements else None
        return lambda i, j: float(emb[i] @ emb[j])
    if similarity != "lexical":
        raise ValueError(f"unknown similarity {similarity!r}; use 'lexical' or 'embedding'")
    return lambda i, j: _lexical_similarity(elements[i].text, elements[j].text)


def build_requirement_set_graph(
        items, template: Template = RUPP_TEMPLATE, extractor=None,
        roles=(Role.SUBJECT, Role.OBJECT), threshold: float = 0.6,
        similarity: str = "lexical",
        embedding_model: str = "prajjwal1/bert-tiny") -> RequirementSetGraph:
    """Parse a requirement set and connect requirements that share similar
    SUBJECT/OBJECT (or other) elements.

    ``items`` accepts the same shapes as ``io_formats``: ``["text", ...]`` or
    ``[(id, text), ...]``. Each item is split into its own atomic
    requirements first (a compound "...shall X and ...shall Y" row becomes
    two), so a "set" can be either one-requirement-per-row or free text.

    Returns a :class:`RequirementSetGraph` with one entry per resulting
    requirement plus the discovered cross-requirement ``connections``.
    """
    parser = RequirementParser(template, extractor)
    req_ids, texts, graphs = [], {}, {}
    elements: list[ElementRef] = []

    counter = 0
    for rid, text in _normalise_items(items):
        counter += 1
        base = rid or f"REQ-{counter}"
        segments = parser.split(text)
        multi = len(segments) > 1
        for i, seg in enumerate(segments, 1):
            display_id = f"{base}-{i}" if multi else base
            while display_id in graphs:                 # keep ids unique
                display_id = f"{display_id}*"
            g = parser.parse(seg, metadata={"id": display_id})
            req_ids.append(display_id)
            texts[display_id] = seg
            graphs[display_id] = g
            for n in g.elements():
                if n.role in roles:
                    elements.append(ElementRef(display_id, n.id, n.role, n.text.strip()))

    score = _score_fn(elements, similarity, embedding_model)
    connections = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            a, b = elements[i], elements[j]
            if a.req_id == b.req_id or a.role is not b.role:
                continue
            s = score(i, j)
            if s >= threshold:
                connections.append(Connection(a, b, a.role, s))
    connections.sort(key=lambda c: -c.score)

    return RequirementSetGraph(req_ids, texts, graphs, connections)
