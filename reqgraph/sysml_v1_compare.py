"""
reqgraph.sysml_v1_compare
==========================

Context-aware semantic comparison of a SysML v1 model (XMI or Turtle) against
a set of natural-language requirements.

Unlike the shallow name-matching approach of the v2 module, this module:

1. Builds a **neighborhood context string** for each model element by BFS
   traversal of the knowledge graph (up to ``context_hops`` hops).  The
   context string concatenates the element's own name + doc + adjacent
   element names + their docs.

2. Scores each (element, requirement) pair using a **confidence formula**:

       confidence = 0.25 * name_score
                  + 0.55 * context_score
                  + 0.20 * satisfaction_bonus

   where ``satisfaction_bonus = 1.0`` if the model already contains a
   satisfy/refine link from the element to a model requirement whose text
   scores ≥ 0.5 against the input requirement text.

3. Produces an **OntologyDiff** comparing the IREB role structure of the
   requirements against the SysML element type/stereotype structure of the
   model, visualised as a Mermaid diagram with two subgraphs and MAPS_TO
   edges.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .core import Role, _xml_escape, _mermaid_label
from .sysml_v1_parser import V1Element, V1Relation, SysMLV1Model
from .templates import RUPP_TEMPLATE, Template


# ---------------------------------------------------------------------------
# Helpers shared with sysml_compare (avoid import cycle)
# ---------------------------------------------------------------------------

import re as _re

_WORD_RE = _re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "its", "their",
    "shall", "should", "must", "will", "can", "be", "to", "of", "and",
    "or", "in", "on", "at", "for", "with", "not", "is", "are", "was",
})


def _norm(s: str) -> str:
    return " ".join(w for w in _WORD_RE.findall(s.lower()) if w not in _STOPWORDS)


def _lexical_sim(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    ta, tb = set(na.split()), set(nb.split())
    jaccard = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    return max(jaccard, ratio)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(element: V1Element, model: SysMLV1Model, hops: int = 2) -> str:
    """Build a neighborhood context string by BFS over model relations."""
    parts = [element.name]
    if element.doc:
        parts.append(element.doc)
    if element.req_text:
        parts.append(element.req_text)

    elem_index: dict[str, V1Element] = {e.xmi_id: e for e in model.elements if e.xmi_id}
    visited = {element.xmi_id}
    frontier = {element.xmi_id}

    for _ in range(hops):
        next_frontier: set[str] = set()
        for eid in frontier:
            for rel in model.relations:
                nid = None
                if rel.source_id == eid and rel.target_id not in visited:
                    nid = rel.target_id
                elif rel.target_id == eid and rel.source_id not in visited:
                    nid = rel.source_id
                if nid:
                    neighbor = elem_index.get(nid)
                    if neighbor:
                        parts.extend([neighbor.name, neighbor.doc or "",
                                      neighbor.req_text or ""])
                        next_frontier.add(nid)
        visited |= next_frontier
        frontier = next_frontier

    return " ".join(p for p in parts if p.strip())


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class V1MatchDetail:
    """One match between a model element and a requirement element."""
    element: V1Element
    req_id: str
    req_text: str
    role: Role
    name_score: float
    context_score: float
    confidence: float
    context_used: str         # the neighborhood string (transparency)
    hops_used: int
    via_satisfaction: bool    # model has satisfy/refine link on this element


@dataclass
class OntologyDiff:
    """Side-by-side ontology comparison: requirements IREB roles vs model types."""
    req_ontology_nodes: list[dict]     # {type, count, instances}
    req_ontology_edges: list[dict]     # {rel, source_type, target_type, count}
    model_ontology_nodes: list[dict]   # {type, stereotype, count}
    model_ontology_edges: list[dict]   # {rel_type, source_type, target_type, count}
    mappings: list[dict]               # {model_type, req_role, mean_confidence, n_matches}

    def to_mermaid(self) -> str:
        """Two-subgraph Mermaid flowchart with MAPS_TO edges."""
        lines = ["flowchart LR"]

        # Requirement ontology subgraph
        lines.append("  subgraph reqont[\"Requirement Ontology\"]")
        for n in self.req_ontology_nodes:
            nid = f"REQ_{n['type']}"
            label = f"{n['type']} ({n['count']})"
            lines.append(f'    {nid}["{_mermaid_label(label)}"]')
        for e in self.req_ontology_edges:
            src = f"REQ_{e['source_type']}"
            tgt = f"REQ_{e['target_type']}"
            lines.append(f'    {src} -->|"{_mermaid_label(e["rel"])}"| {tgt}')
        lines.append("  end")

        # Model ontology subgraph
        lines.append("  subgraph modelont[\"Model Ontology\"]")
        for n in self.model_ontology_nodes:
            nid = f"MDL_{n['type']}"
            label = f"{n['type']} ({n['count']})"
            lines.append(f'    {nid}["{_mermaid_label(label)}"]')
        for e in self.model_ontology_edges:
            src = f"MDL_{e['source_type']}"
            tgt = f"MDL_{e['target_type']}"
            lines.append(f'    {src} -->|"{_mermaid_label(e["rel_type"])}"| {tgt}')
        lines.append("  end")

        # Mapping edges
        for m in self.mappings:
            mtype = f"MDL_{m['model_type']}"
            rrole = f"REQ_{m['req_role']}"
            conf = f"{m['mean_confidence']:.2f}"
            lines.append(f'  {mtype} -.->|"{conf}"| {rrole}')

        return "\n".join(lines)

    def to_graphml(self) -> str:
        """Combined bipartite ontology GraphML."""
        ns = "http://graphml.graphdrawing.org/xmlns"
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<graphml xmlns="{ns}">',
            '  <key id="ont" for="node" attr.name="ont" attr.type="string"/>',
            '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
            '  <key id="count" for="node" attr.name="count" attr.type="int"/>',
            '  <key id="rel" for="edge" attr.name="rel" attr.type="string"/>',
            '  <key id="confidence" for="edge" attr.name="confidence" attr.type="double"/>',
            '  <graph id="ontology-diff" edgedefault="directed">',
        ]
        for n in self.req_ontology_nodes:
            nid = _xml_escape(f"REQ_{n['type']}")
            lines += [f'    <node id="{nid}">',
                      f'      <data key="ont">requirement</data>',
                      f'      <data key="label">{_xml_escape(n["type"])}</data>',
                      f'      <data key="count">{n["count"]}</data>',
                      '    </node>']
        for n in self.model_ontology_nodes:
            nid = _xml_escape(f"MDL_{n['type']}")
            lines += [f'    <node id="{nid}">',
                      f'      <data key="ont">model</data>',
                      f'      <data key="label">{_xml_escape(n["type"])}</data>',
                      f'      <data key="count">{n["count"]}</data>',
                      '    </node>']
        edge_i = 0
        for e in self.req_ontology_edges:
            src, tgt = _xml_escape(f"REQ_{e['source_type']}"), _xml_escape(f"REQ_{e['target_type']}")
            lines += [f'    <edge id="e{edge_i}" source="{src}" target="{tgt}">',
                      f'      <data key="rel">{_xml_escape(e["rel"])}</data>',
                      '    </edge>']
            edge_i += 1
        for m in self.mappings:
            src = _xml_escape(f"MDL_{m['model_type']}")
            tgt = _xml_escape(f"REQ_{m['req_role']}")
            conf = f"{m['mean_confidence']:.4f}"
            lines += [f'    <edge id="e{edge_i}" source="{src}" target="{tgt}">',
                      f'      <data key="rel">MAPS_TO</data>',
                      f'      <data key="confidence">{conf}</data>',
                      '    </edge>']
            edge_i += 1
        lines += ['  </graph>', '</graphml>']
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "req_ontology_nodes": self.req_ontology_nodes,
            "req_ontology_edges": self.req_ontology_edges,
            "model_ontology_nodes": self.model_ontology_nodes,
            "model_ontology_edges": self.model_ontology_edges,
            "mappings": self.mappings,
        }


@dataclass
class V1ComparisonReport:
    """Results of a SysML v1 model ↔ requirement set comparison."""
    model_coverage: float
    req_coverage: float
    semantic_match: float
    role_breakdown: dict
    matches: list[V1MatchDetail]
    unmatched_model: list[V1Element]
    unmatched_reqs: list[tuple]      # (req_id, role_val, text)
    n_model_elements: int
    n_req_elements: int
    warnings: list[str]
    ontology_diff: OntologyDiff | None = None

    def to_dict(self) -> dict:
        return {
            "model_coverage": round(self.model_coverage, 4),
            "req_coverage": round(self.req_coverage, 4),
            "semantic_match": round(self.semantic_match, 4),
            "n_model_elements": self.n_model_elements,
            "n_req_elements": self.n_req_elements,
            "role_breakdown": self.role_breakdown,
            "matches": [
                {
                    "element_name": m.element.name,
                    "element_type": m.element.element_type,
                    "stereotype": m.element.stereotype,
                    "package": m.element.package,
                    "req_id": m.req_id,
                    "req_text": m.req_text,
                    "role": m.role.value,
                    "name_score": round(m.name_score, 4),
                    "context_score": round(m.context_score, 4),
                    "confidence": round(m.confidence, 4),
                    "via_satisfaction": m.via_satisfaction,
                    "context_used": m.context_used,
                }
                for m in self.matches
            ],
            "unmatched_model": [
                {"name": e.name, "element_type": e.element_type,
                 "stereotype": e.stereotype, "package": e.package,
                 "role": e.role.value}
                for e in self.unmatched_model
            ],
            "unmatched_reqs": [
                {"req_id": rid, "role": rv, "text": txt}
                for rid, rv, txt in self.unmatched_reqs
            ],
            "warnings": self.warnings,
            "ontology_diff": self.ontology_diff.to_dict() if self.ontology_diff else None,
        }

    def to_graphml(self) -> str:
        """Bipartite match graph + context hop edges."""
        ns = "http://graphml.graphdrawing.org/xmlns"
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<graphml xmlns="{ns}">',
            '  <key id="node_type"    for="node" attr.name="node_type"    attr.type="string"/>',
            '  <key id="name"         for="node" attr.name="name"         attr.type="string"/>',
            '  <key id="role"         for="node" attr.name="role"         attr.type="string"/>',
            '  <key id="req_id"       for="node" attr.name="req_id"       attr.type="string"/>',
            '  <key id="text"         for="node" attr.name="text"         attr.type="string"/>',
            '  <key id="stereotype"   for="node" attr.name="stereotype"   attr.type="string"/>',
            '  <key id="rel_type"     for="edge" attr.name="rel_type"     attr.type="string"/>',
            '  <key id="confidence"   for="edge" attr.name="confidence"   attr.type="double"/>',
            '  <key id="name_score"   for="edge" attr.name="name_score"   attr.type="double"/>',
            '  <key id="ctx_score"    for="edge" attr.name="ctx_score"    attr.type="double"/>',
            '  <graph id="v1-compare" edgedefault="directed">',
        ]

        # Deduplicate model nodes by name
        seen_model: dict[str, V1Element] = {}
        for m in self.matches:
            if m.element.name not in seen_model:
                seen_model[m.element.name] = m.element
        for e in self.unmatched_model:
            if e.name not in seen_model:
                seen_model[e.name] = e

        # Req element nodes
        seen_req: dict[str, tuple] = {}
        for m in self.matches:
            key = f"{m.req_id}__{m.req_text[:40]}"
            if key not in seen_req:
                seen_req[key] = (m.req_id, m.req_text, m.role)
        for rid, rv, txt in self.unmatched_reqs:
            key = f"{rid}__{txt[:40]}"
            if key not in seen_req:
                seen_req[key] = (rid, txt, Role(rv) if isinstance(rv, str) else rv)

        # Emit model nodes
        for name, e in seen_model.items():
            nid = _xml_escape(f"model_{name}")
            lines += [f'    <node id="{nid}">',
                      f'      <data key="node_type">MODEL_ELEMENT</data>',
                      f'      <data key="name">{_xml_escape(e.name)}</data>',
                      f'      <data key="role">{_xml_escape(e.role.value)}</data>',
                      f'      <data key="stereotype">{_xml_escape(e.stereotype)}</data>',
                      '    </node>']

        # Emit req nodes
        for key, (rid, txt, role) in seen_req.items():
            nid = _xml_escape(f"req_{key}")
            lines += [f'    <node id="{nid}">',
                      f'      <data key="node_type">REQ_ELEMENT</data>',
                      f'      <data key="req_id">{_xml_escape(rid)}</data>',
                      f'      <data key="text">{_xml_escape(txt)}</data>',
                      f'      <data key="role">{_xml_escape(role.value)}</data>',
                      '    </node>']

        # Emit match edges
        for i, m in enumerate(self.matches):
            src = _xml_escape(f"model_{m.element.name}")
            key = f"{m.req_id}__{m.req_text[:40]}"
            tgt = _xml_escape(f"req_{key}")
            lines += [f'    <edge id="m{i}" source="{src}" target="{tgt}">',
                      f'      <data key="rel_type">MATCHES</data>',
                      f'      <data key="confidence">{m.confidence:.4f}</data>',
                      f'      <data key="name_score">{m.name_score:.4f}</data>',
                      f'      <data key="ctx_score">{m.context_score:.4f}</data>',
                      '    </edge>']

        lines += ['  </graph>', '</graphml>']
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Flowchart: model elements → matched requirement elements."""
        lines = ["flowchart LR"]
        seen_model: set[str] = set()
        seen_req: set[str] = set()
        for m in self.matches[:30]:  # cap at 30 for readability
            mn = f"m_{_re.sub(r'[^A-Za-z0-9]', '_', m.element.name)}"
            rn = f"r_{_re.sub(r'[^A-Za-z0-9]', '_', m.req_id)}_{m.req_text[:20].replace(' ', '_')}"
            rn = _re.sub(r"[^A-Za-z0-9_]", "_", rn)
            if mn not in seen_model:
                label = f"{m.element.name} [{m.element.stereotype or m.element.element_type}]"
                lines.append(f'    {mn}["{_mermaid_label(label)}"]')
                seen_model.add(mn)
            if rn not in seen_req:
                txt = m.req_text if len(m.req_text) <= 40 else m.req_text[:37] + "..."
                lines.append(f'    {rn}["{_mermaid_label(m.req_id)}: {_mermaid_label(txt)}"]')
                seen_req.add(rn)
            lines.append(
                f'    {mn} -->|"{m.role.value} {m.confidence:.2f}"| {rn}')
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ontology diff builder
# ---------------------------------------------------------------------------

def ontology_diff(model: SysMLV1Model, rsg, matches: list[V1MatchDetail]) -> OntologyDiff:
    """Build a side-by-side IREB vs SysML ontology comparison.

    *rsg* is a ``RequirementSetGraph`` (may be None if unavailable).
    """
    # Requirement ontology nodes (IREB roles)
    req_role_counts: dict[str, int] = {}
    req_role_instances: dict[str, list[str]] = {}
    if rsg is not None:
        for rid in rsg.req_ids:
            for n in rsg.graphs[rid].elements():
                rv = n.role.value
                req_role_counts[rv] = req_role_counts.get(rv, 0) + 1
                req_role_instances.setdefault(rv, []).append(n.text.strip()[:50])

    req_ont_nodes = [
        {"type": rv, "count": cnt, "instances": req_role_instances.get(rv, [])}
        for rv, cnt in sorted(req_role_counts.items())
    ]

    # Standard IREB role graph edges
    _REQ_ROLE_EDGES = [
        {"rel": "HAS_CONDITION", "source_type": "CONDITION", "target_type": "SUBJECT", "count": 0},
        {"rel": "HAS_SUBJECT", "source_type": "SUBJECT", "target_type": "PROCESS", "count": 0},
        {"rel": "ACTS_ON", "source_type": "PROCESS", "target_type": "OBJECT", "count": 0},
        {"rel": "CONSTRAINED_BY", "source_type": "PROCESS", "target_type": "CONSTRAINT", "count": 0},
    ]
    req_ont_edges = [e for e in _REQ_ROLE_EDGES
                     if e["source_type"] in req_role_counts and e["target_type"] in req_role_counts]

    # Model ontology nodes (by stereotype/element_type)
    model_type_counts: dict[str, int] = {}
    for e in model.elements:
        key = e.stereotype if e.stereotype else e.element_type.split(":")[-1]
        model_type_counts[key] = model_type_counts.get(key, 0) + 1

    model_ont_nodes = [
        {"type": t, "stereotype": t, "count": c}
        for t, c in sorted(model_type_counts.items())
    ]

    # Model ontology edges (from relations)
    rel_type_counts: dict[tuple, int] = {}
    elem_index = {e.xmi_id: e for e in model.elements if e.xmi_id}
    for r in model.relations:
        src_e = elem_index.get(r.source_id)
        tgt_e = elem_index.get(r.target_id)
        if src_e and tgt_e:
            src_type = src_e.stereotype or src_e.element_type.split(":")[-1]
            tgt_type = tgt_e.stereotype or tgt_e.element_type.split(":")[-1]
            key = (r.rel_type, src_type, tgt_type)
            rel_type_counts[key] = rel_type_counts.get(key, 0) + 1

    model_ont_edges = [
        {"rel_type": k[0], "source_type": k[1], "target_type": k[2], "count": v}
        for k, v in sorted(rel_type_counts.items())
    ]

    # Mapping layer: model_type → req_role, weighted by confidence
    mapping_scores: dict[tuple, list[float]] = {}
    for m in matches:
        model_type = m.element.stereotype or m.element.element_type.split(":")[-1]
        req_role = m.role.value
        key = (model_type, req_role)
        mapping_scores.setdefault(key, []).append(m.confidence)

    mappings = [
        {
            "model_type": k[0],
            "req_role": k[1],
            "mean_confidence": round(sum(v) / len(v), 4),
            "n_matches": len(v),
        }
        for k, v in sorted(mapping_scores.items())
    ]

    return OntologyDiff(
        req_ontology_nodes=req_ont_nodes,
        req_ontology_edges=req_ont_edges,
        model_ontology_nodes=model_ont_nodes,
        model_ontology_edges=model_ont_edges,
        mappings=mappings,
    )


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_v1(
    model: SysMLV1Model,
    items,
    roles: tuple = (Role.SUBJECT, Role.PROCESS, Role.OBJECT, Role.CONDITION),
    threshold: float = 0.5,
    similarity: str = "lexical",
    embedding_model: str = "prajjwal1/bert-tiny",
    context_hops: int = 2,
    template: Template = RUPP_TEMPLATE,
    extractor=None,
) -> V1ComparisonReport:
    """Compare a SysML v1 model against a requirement set.

    Parameters
    ----------
    model:
        Parsed ``SysMLV1Model`` (from ``parse_sysml_v1`` or ``read_sysml_v1``).
    items:
        Requirements: plain strings, ``(id, text)`` 2-tuples, or
        ``(id, text, meta)`` 3-tuples.
    roles:
        IREB roles to include in the comparison.
    threshold:
        Confidence cut-off for a match (default 0.5, lower than v2 because
        context scoring is richer).
    similarity:
        ``"lexical"`` (default, no deps) or ``"embedding"``
        (requires torch + transformers).
    context_hops:
        BFS depth for neighborhood context building (default 2).
    """
    from .corpus import build_requirement_set_graph

    warnings: list[str] = []

    if not model.elements:
        warnings.append("SysML v1 model contains no parseable elements")

    # Build requirement set graph (threshold=0.0 → keep all elements)
    rsg = build_requirement_set_graph(
        items, template=template, extractor=extractor,
        roles=roles, threshold=0.0, similarity="lexical")

    # Collect all requirement elements (req_id, role, text)
    all_req_elems: list[tuple] = []
    for rid in rsg.req_ids:
        for n in rsg.graphs[rid].elements():
            if n.role in roles:
                all_req_elems.append((rid, n.role, n.text.strip()))

    if not all_req_elems:
        warnings.append("no requirement elements found for the specified roles")

    # Score function
    if similarity == "embedding":
        try:
            from .nlp import RequirementAnalyzer
            _analyzer = RequirementAnalyzer(model_name=embedding_model)
            def score_fn(a: str, b: str) -> float:
                vecs = _analyzer.embed([a, b])
                va, vb = vecs[0], vecs[1]
                dot = sum(x * y for x, y in zip(va, vb))
                return max(0.0, min(1.0, dot))
        except ImportError:
            warnings.append("embedding similarity requires torch+transformers; "
                            "falling back to lexical")
            score_fn = _lexical_sim
    else:
        score_fn = _lexical_sim

    # Pre-build contexts for all model elements
    model_elems = [e for e in model.elements if e.role in roles]
    contexts: dict[str, str] = {}
    for e in model_elems:
        contexts[e.xmi_id or e.name] = _build_context(e, model, hops=context_hops)

    # Pre-compute satisfaction links: set of (element_xmi_id)
    # An element has a satisfaction bonus if it has a satisfy/refine relation
    # to a model requirement whose req_text scores ≥ 0.5 against the target.
    req_elements_in_model = {e.xmi_id: e for e in model.elements
                              if e.stereotype.lower() in ("requirement",)
                              and e.req_text}
    satisfy_links: dict[str, set[str]] = {}  # source_id → set of req xmi_ids
    for r in model.relations:
        if r.rel_type in ("satisfy", "refine"):
            satisfy_links.setdefault(r.source_id, set()).add(r.target_id)

    # Per-role containers
    role_model: dict[Role, list[V1Element]] = {r: [] for r in roles}
    role_req: dict[Role, list[tuple]] = {r: [] for r in roles}

    for e in model_elems:
        if e.role in role_model:
            role_model[e.role].append(e)

    for rid, role, txt in all_req_elems:
        if role in role_req:
            role_req[role].append((rid, txt))

    all_matches: list[V1MatchDetail] = []
    matched_model_ids: set[str] = set()
    matched_req_keys: set[tuple] = set()

    role_breakdown: dict[str, dict] = {}

    for role in roles:
        m_elems = role_model[role]
        r_elems = role_req[role]

        if not m_elems and not r_elems:
            role_breakdown[role.value] = {
                "model_coverage": 0.0, "req_coverage": 0.0,
                "score": 0.0, "n_model": 0, "n_req": 0,
            }
            continue

        if not m_elems or not r_elems:
            role_breakdown[role.value] = {
                "model_coverage": 0.0 if not m_elems else 0.0,
                "req_coverage": 0.0,
                "score": 0.0,
                "n_model": len(m_elems),
                "n_req": len(r_elems),
            }
            continue

        # Score matrix
        for e in m_elems:
            ctx = contexts.get(e.xmi_id or e.name, e.name)
            best_conf = 0.0
            best_match: V1MatchDetail | None = None

            # Check satisfaction bonus
            sat_ids = satisfy_links.get(e.xmi_id, set())

            for req_id, req_txt in r_elems:
                name_s = score_fn(e.name, req_txt)
                ctx_s = score_fn(ctx, req_txt)

                # Satisfaction bonus: check if any sat-linked req scores high
                sat_bonus = 0.0
                for sat_req_id in sat_ids:
                    sat_req = req_elements_in_model.get(sat_req_id)
                    if sat_req and score_fn(sat_req.req_text, req_txt) >= 0.5:
                        sat_bonus = 1.0
                        break

                conf = 0.25 * name_s + 0.55 * ctx_s + 0.20 * sat_bonus
                if conf > best_conf:
                    best_conf = conf
                    best_match = V1MatchDetail(
                        element=e,
                        req_id=req_id,
                        req_text=req_txt,
                        role=role,
                        name_score=name_s,
                        context_score=ctx_s,
                        confidence=conf,
                        context_used=ctx,
                        hops_used=context_hops,
                        via_satisfaction=(sat_bonus > 0.0),
                    )

            if best_match and best_match.confidence >= threshold:
                all_matches.append(best_match)
                matched_model_ids.add(e.xmi_id or e.name)

        for req_id, req_txt in r_elems:
            best_conf = 0.0
            for e in m_elems:
                ctx = contexts.get(e.xmi_id or e.name, e.name)
                name_s = score_fn(e.name, req_txt)
                ctx_s = score_fn(ctx, req_txt)
                sat_ids = satisfy_links.get(e.xmi_id, set())
                sat_bonus = 0.0
                for sat_req_id in sat_ids:
                    sat_req = req_elements_in_model.get(sat_req_id)
                    if sat_req and score_fn(sat_req.req_text, req_txt) >= 0.5:
                        sat_bonus = 1.0
                        break
                conf = 0.25 * name_s + 0.55 * ctx_s + 0.20 * sat_bonus
                if conf > best_conf:
                    best_conf = conf
            if best_conf >= threshold:
                matched_req_keys.add((req_id, req_txt))

        # Role-level coverage
        mc = (sum(1 for e in m_elems
                  if (e.xmi_id or e.name) in matched_model_ids) / len(m_elems)
              if m_elems else 0.0)
        rc = (sum(1 for (rid, txt) in r_elems
                  if (rid, txt) in matched_req_keys) / len(r_elems)
              if r_elems else 0.0)
        s = (2 * mc * rc / (mc + rc)) if (mc + rc) > 0 else 0.0

        role_breakdown[role.value] = {
            "model_coverage": round(mc, 4),
            "req_coverage": round(rc, 4),
            "score": round(s, 4),
            "n_model": len(m_elems),
            "n_req": len(r_elems),
        }

    # Overall metrics
    n_model = len(model_elems)
    n_req = len(all_req_elems)
    overall_mc = (len(matched_model_ids) / n_model) if n_model else 0.0
    overall_rc = (len(matched_req_keys) / n_req) if n_req else 0.0
    semantic_match = ((2 * overall_mc * overall_rc / (overall_mc + overall_rc))
                      if (overall_mc + overall_rc) > 0 else 0.0)

    # Unmatched
    matched_id_set = {m.element.xmi_id or m.element.name for m in all_matches}
    unmatched_model = [e for e in model_elems
                       if (e.xmi_id or e.name) not in matched_id_set]
    unmatched_reqs = [
        (rid, role.value, txt)
        for rid, role, txt in all_req_elems
        if (rid, txt) not in matched_req_keys
    ]

    # Sort matches by confidence descending
    all_matches.sort(key=lambda m: m.confidence, reverse=True)

    # Ontology diff
    ont_diff = ontology_diff(model, rsg, all_matches)

    return V1ComparisonReport(
        model_coverage=round(overall_mc, 4),
        req_coverage=round(overall_rc, 4),
        semantic_match=round(semantic_match, 4),
        role_breakdown=role_breakdown,
        matches=all_matches,
        unmatched_model=unmatched_model,
        unmatched_reqs=unmatched_reqs,
        n_model_elements=n_model,
        n_req_elements=n_req,
        warnings=warnings,
        ontology_diff=ont_diff,
    )
