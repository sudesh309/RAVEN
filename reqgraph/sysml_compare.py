"""
reqgraph.sysml_compare
=======================

Semantic comparison of a SysML v2 model against a set of natural-language
requirements.

The comparison maps SysML structural/behavioural elements onto the same
IREB semantic roles that the requirement parser extracts (SUBJECT, PROCESS,
OBJECT, CONDITION, …).  For each role bucket, every model element is paired
with every requirement element of the same role and their similarity is
scored.  The result is:

* **Model coverage** — what fraction of model elements appear in requirements
  (≥ threshold).  Low model coverage → parts of the architecture are not
  traced to requirements.
* **Requirement coverage** — what fraction of requirement semantic elements
  appear in the model.  Low req coverage → requirements are not implemented.
* **Semantic match** — F1-like harmonic mean of model and req coverage.
  Ranges from 0 (no overlap) to 1.0 (perfect bidirectional coverage).

Role breakdown and per-match details are also returned so callers can inspect
which specific elements are matched, unmatched, or surprising.

Usage::

    from reqgraph.sysml_parser import read_sysml
    from reqgraph.sysml_compare import compare

    model = read_sysml("automotive.sysml")
    reqs  = [
        ("R1", "The vehicle shall respond to brake commands."),
        ("R2", "The braking system shall decelerate at 5 m/s²."),
    ]
    report = compare(model, reqs)
    print(f"Semantic match: {report.semantic_match:.1%}")
    print(f"Model coverage: {report.model_coverage:.1%}")
    print(f"Req coverage:   {report.req_coverage:.1%}")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .core import Role, _xml_escape, _mermaid_label
from .sysml_parser import SysMLElement, SysMLModel
from .templates import RUPP_TEMPLATE, Template


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MatchDetail:
    """A similarity match between a SysML element and a requirement element."""
    sysml_element: SysMLElement  # model element
    req_id: str                  # source requirement id
    req_text: str                # matched element text from the requirement
    role: Role
    score: float                 # similarity score [0, 1]


@dataclass
class ComparisonReport:
    """Results of a SysML model ↔ requirement set comparison.

    All float scores range from 0.0 (no match) to 1.0 (perfect coverage).
    """
    #: Fraction of model elements matched to at least one requirement element.
    model_coverage: float
    #: Fraction of requirement elements matched to at least one model element.
    req_coverage: float
    #: F1-like harmonic mean of model_coverage and req_coverage.
    semantic_match: float
    #: Per-role breakdown {role_name: {model_coverage, req_coverage, score, n_model, n_req}}.
    role_breakdown: dict
    #: All matches above threshold.
    matches: list[MatchDetail]
    #: Model elements with no matching requirement element.
    unmatched_model: list[SysMLElement]
    #: (req_id, role_value, element_text) for req elements with no model match.
    unmatched_reqs: list[tuple]
    #: Number of model elements considered (in the requested roles).
    n_model_elements: int
    #: Number of requirement elements considered (in the requested roles).
    n_req_elements: int
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of the report."""
        return {
            "semantic_match": round(self.semantic_match, 4),
            "model_coverage":  round(self.model_coverage, 4),
            "req_coverage":    round(self.req_coverage, 4),
            "n_model_elements": self.n_model_elements,
            "n_req_elements":   self.n_req_elements,
            "role_breakdown": self.role_breakdown,
            "matches": [
                {"model_name": m.sysml_element.name,
                 "model_role": m.role.value,
                 "model_element_type": m.sysml_element.element_type,
                 "model_package": m.sysml_element.package,
                 "req_id": m.req_id,
                 "req_text": m.req_text,
                 "score": m.score}
                for m in self.matches
            ],
            "unmatched_model": [
                {"name": e.name, "role": e.role.value,
                 "element_type": e.element_type, "package": e.package}
                for e in self.unmatched_model
            ],
            "unmatched_reqs": [
                {"req_id": req_id, "role": role_val, "text": txt}
                for req_id, role_val, txt in self.unmatched_reqs
            ],
            "warnings": self.warnings,
        }

    # ------------------------------------------------------------------
    def to_graphml(self) -> str:
        """Bipartite GraphML: MODEL nodes on the left, REQ_ELEMENT nodes on
        the right, MATCHES edges connecting them.  Unmatched elements appear
        as isolated nodes with ``unmatched="true"``."""
        ns = "http://graphml.graphdrawing.org/xmlns"
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<graphml xmlns="{ns}">',
            '  <key id="node_type"    for="node" attr.name="node_type"    attr.type="string"/>',
            '  <key id="name"         for="node" attr.name="name"         attr.type="string"/>',
            '  <key id="role"         for="node" attr.name="role"         attr.type="string"/>',
            '  <key id="element_type" for="node" attr.name="element_type" attr.type="string"/>',
            '  <key id="package"      for="node" attr.name="package"      attr.type="string"/>',
            '  <key id="req_id"       for="node" attr.name="req_id"       attr.type="string"/>',
            '  <key id="text"         for="node" attr.name="text"         attr.type="string"/>',
            '  <key id="unmatched"    for="node" attr.name="unmatched"    attr.type="string"/>',
            '  <key id="rel"          for="edge" attr.name="rel"          attr.type="string"/>',
            '  <key id="score"        for="edge" attr.name="score"        attr.type="double"/>',
            '  <key id="role_e"       for="edge" attr.name="role"         attr.type="string"/>',
            '  <graph id="sysml-req-comparison" edgedefault="undirected">',
        ]
        edge_ctr = 0

        # Build node id sets so unmatched detection is easy
        matched_model_names: set[str] = {m.sysml_element.name for m in self.matches}
        matched_req_keys: set[tuple] = {(m.req_id, m.role.value, m.req_text)
                                        for m in self.matches}

        # MODEL nodes
        model_ids: dict[str, str] = {}  # name → gml node id
        seen: set[str] = set()
        # Collect matched elements by name (dedup) then append unmatched
        _matched_by_name: dict[str, SysMLElement] = {}
        for m in self.matches:
            if m.sysml_element.name not in _matched_by_name:
                _matched_by_name[m.sysml_element.name] = m.sysml_element
        _all_model = list(_matched_by_name.values()) + self.unmatched_model
        for e in _all_model:
            if e.name in seen:
                continue
            seen.add(e.name)
            nid = f"m_{_safe(e.name)}"
            # ensure uniqueness
            orig = nid
            ctr = 0
            while nid in model_ids.values():
                ctr += 1
                nid = f"{orig}_{ctr}"
            model_ids[e.name] = nid
            unmatched_val = "false" if e.name in matched_model_names else "true"
            lines += [
                f'    <node id="{_xml_escape(nid)}">',
                f'      <data key="node_type">MODEL</data>',
                f'      <data key="name">{_xml_escape(e.name)}</data>',
                f'      <data key="role">{_xml_escape(e.role.value)}</data>',
                f'      <data key="element_type">{_xml_escape(e.element_type)}</data>',
                f'      <data key="package">{_xml_escape(e.package)}</data>',
                f'      <data key="unmatched">{unmatched_val}</data>',
                '    </node>',
            ]

        # REQ_ELEMENT nodes
        req_ids: dict[tuple, str] = {}  # (req_id, role, text) → gml node id
        seen_req: set[tuple] = set()
        for m in self.matches:
            key = (m.req_id, m.role.value, m.req_text)
            if key not in seen_req:
                seen_req.add(key)
                nid = f"r_{_safe(m.req_id)}_{_safe(m.role.value)}_{_safe(m.req_text[:20])}"
                orig = nid
                ctr = 0
                while nid in req_ids.values():
                    ctr += 1
                    nid = f"{orig}_{ctr}"
                req_ids[key] = nid
                lines += [
                    f'    <node id="{_xml_escape(nid)}">',
                    f'      <data key="node_type">REQ_ELEMENT</data>',
                    f'      <data key="req_id">{_xml_escape(m.req_id)}</data>',
                    f'      <data key="role">{_xml_escape(m.role.value)}</data>',
                    f'      <data key="text">{_xml_escape(m.req_text)}</data>',
                    f'      <data key="unmatched">false</data>',
                    '    </node>',
                ]
        for req_id, role_val, txt in self.unmatched_reqs:
            key = (req_id, role_val, txt)
            if key not in seen_req:
                seen_req.add(key)
                nid = f"r_{_safe(req_id)}_{_safe(role_val)}_{_safe(txt[:20])}"
                orig = nid
                ctr = 0
                while nid in req_ids.values():
                    ctr += 1
                    nid = f"{orig}_{ctr}"
                req_ids[key] = nid
                lines += [
                    f'    <node id="{_xml_escape(nid)}">',
                    f'      <data key="node_type">REQ_ELEMENT</data>',
                    f'      <data key="req_id">{_xml_escape(req_id)}</data>',
                    f'      <data key="role">{_xml_escape(role_val)}</data>',
                    f'      <data key="text">{_xml_escape(txt)}</data>',
                    f'      <data key="unmatched">true</data>',
                    '    </node>',
                ]

        # MATCHES edges
        for m in self.matches:
            src = model_ids.get(m.sysml_element.name, "")
            tgt = req_ids.get((m.req_id, m.role.value, m.req_text), "")
            if not src or not tgt:
                continue
            lines += [
                f'    <edge id="e{edge_ctr}" '
                f'source="{_xml_escape(src)}" target="{_xml_escape(tgt)}">',
                f'      <data key="rel">MATCHES</data>',
                f'      <data key="score">{m.score:.4f}</data>',
                f'      <data key="role_e">{_xml_escape(m.role.value)}</data>',
                '    </edge>',
            ]
            edge_ctr += 1

        lines += ['  </graph>', '</graphml>']
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def to_mermaid(self) -> str:
        """Flowchart showing model elements → matched requirement elements."""
        lines = ["flowchart LR"]
        model_seen: set[str] = set()
        req_seen: set[str] = set()

        def model_node_id(e: SysMLElement) -> str:
            return f"m_{_safe(e.name)}"

        def req_node_id(req_id: str, role_val: str, text: str) -> str:
            return f"r_{_safe(req_id)}_{_safe(role_val)}_{_safe(text[:15])}"

        for m in self.matches:
            mid = model_node_id(m.sysml_element)
            rid = req_node_id(m.req_id, m.role.value, m.req_text)
            if mid not in model_seen:
                model_seen.add(mid)
                lbl = f"[{m.sysml_element.element_type}] {m.sysml_element.name}"
                lines.append(f'  {mid}["{_mermaid_label(lbl)}"]')
            if rid not in req_seen:
                req_seen.add(rid)
                lbl = f"[{m.req_id}/{m.role.value}] {m.req_text}"
                if len(lbl) > 55:
                    lbl = lbl[:52] + "…"
                lines.append(f'  {rid}(("{_mermaid_label(lbl)}"))')
            lines.append(
                f'  {mid} -->|"{_mermaid_label(m.role.value)} {m.score:.2f}"| {rid}')

        for e in self.unmatched_model:
            mid = model_node_id(e)
            if mid not in model_seen:
                model_seen.add(mid)
                lbl = f"[{e.element_type}] {e.name}"
                lines.append(f'  {mid}["{_mermaid_label(lbl)}"]:::unmatched')

        if model_seen or req_seen:
            lines.append("  classDef unmatched fill:#FCEBEB,stroke:#A32D2D")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------

def compare(
    model: SysMLModel,
    items,
    roles: tuple = (Role.SUBJECT, Role.PROCESS, Role.OBJECT, Role.CONDITION),
    threshold: float = 0.6,
    similarity: str = "lexical",
    embedding_model: str = "prajjwal1/bert-tiny",
    template: Template = RUPP_TEMPLATE,
    extractor=None,
) -> ComparisonReport:
    """Compare a :class:`~reqgraph.sysml_parser.SysMLModel` against a set of
    natural-language requirements.

    Parameters
    ----------
    model:
        The parsed SysML v2 model (from :func:`~reqgraph.sysml_parser.parse_sysml`
        or :func:`~reqgraph.sysml_parser.read_sysml`).
    items:
        Requirements as strings, ``(id, text)`` tuples, or
        ``(id, text, metadata_dict)`` triples — the same shapes accepted by
        :func:`~reqgraph.corpus.build_requirement_set_graph`.
    roles:
        Semantic roles to compare.  Defaults to SUBJECT, PROCESS, OBJECT,
        CONDITION.  Must be a subset of :class:`~reqgraph.core.Role`.
    threshold:
        Similarity cut-off for a match, 0–1 (default 0.6).
    similarity:
        ``"lexical"`` (fast, no deps) or ``"embedding"`` (BERT cosine;
        requires ``torch`` + ``transformers``).
    embedding_model:
        HuggingFace model name used when ``similarity="embedding"``.
    template:
        Requirement parsing template (default RUPP / IREB-Rupp).
    extractor:
        Extraction backend instance.  ``None`` → uses
        :class:`~reqgraph.extractors.RuleExtractor`.

    Returns
    -------
    ComparisonReport
    """
    from .corpus import build_requirement_set_graph, ElementRef, _lexical_similarity

    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Parse requirements and collect ElementRef objects per role.
    #    Use threshold=0.0 so ALL elements are retained; we do our own
    #    threshold filtering below.
    # ------------------------------------------------------------------
    try:
        rsg = build_requirement_set_graph(
            items, template=template, extractor=extractor,
            roles=roles, threshold=0.0, similarity="lexical")
    except ImportError as exc:
        from .errors import ReqGraphError
        raise ReqGraphError(
            f"compare() failed to build the requirement graph: {exc}") from exc

    # Collect requirement elements per role
    req_elems_by_role: dict[Role, list[ElementRef]] = {}
    for rid in rsg.req_ids:
        g = rsg.graphs[rid]
        for n in g.elements():
            if n.role in roles:
                ref = ElementRef(req_id=rid, node_id=n.id,
                                 role=n.role, text=n.text.strip())
                req_elems_by_role.setdefault(n.role, []).append(ref)

    # ------------------------------------------------------------------
    # 2. Collect model elements per role.
    # ------------------------------------------------------------------
    model_elems_by_role: dict[Role, list[SysMLElement]] = {}
    for e in model.elements:
        if e.role in roles:
            model_elems_by_role.setdefault(e.role, []).append(e)

    # ------------------------------------------------------------------
    # 3. Embedding setup (if needed).
    # ------------------------------------------------------------------
    embeddings: dict | None = None
    if similarity == "embedding":
        try:
            from .nlp import RequirementAnalyzer
            # Gather all texts that will be scored to build one embedding matrix
            all_texts = (
                [e.name for role in roles
                 for e in model_elems_by_role.get(role, [])]
                + [ref.text for role in roles
                   for ref in req_elems_by_role.get(role, [])]
            )
            if all_texts:
                az = RequirementAnalyzer(model_name=embedding_model)
                emb_matrix = az.embed(all_texts)
                # Build lookup: text → embedding row index
                embeddings = {"texts": all_texts, "matrix": emb_matrix}
        except ImportError:
            warnings.append(
                "embedding similarity needs PyTorch + transformers; "
                "falling back to lexical")
            similarity = "lexical"

    def _score(text_a: str, text_b: str) -> float:
        if similarity == "embedding" and embeddings is not None:
            texts = embeddings["texts"]
            mat = embeddings["matrix"]
            try:
                ia = texts.index(text_a)
                ib = texts.index(text_b)
                return float(mat[ia] @ mat[ib])
            except ValueError:
                pass
        return _lexical_similarity(text_a, text_b)

    # ------------------------------------------------------------------
    # 4. Per-role comparison.
    # ------------------------------------------------------------------
    all_matches: list[MatchDetail] = []
    all_unmatched_model: list[SysMLElement] = []
    all_unmatched_reqs: list[tuple] = []
    role_breakdown: dict = {}

    total_model = 0
    total_model_matched = 0
    total_req = 0
    total_req_matched = 0

    for role in roles:
        m_elems = model_elems_by_role.get(role, [])
        r_elems = req_elems_by_role.get(role, [])

        total_model += len(m_elems)
        total_req += len(r_elems)

        if not m_elems and not r_elems:
            continue

        # Pairwise score matrix: rows = model elements, cols = req elements
        score_matrix: list[list[float]] = []
        for me in m_elems:
            row = [_score(me.name, re_.text) for re_ in r_elems]
            score_matrix.append(row)

        # Per-model-element: best match
        matched_model = []
        for i, me in enumerate(m_elems):
            if not r_elems:
                matched_model.append(False)
                all_unmatched_model.append(me)
                continue
            best_j = max(range(len(r_elems)), key=lambda j: score_matrix[i][j])
            best_score = score_matrix[i][best_j]
            if best_score >= threshold:
                matched_model.append(True)
                all_matches.append(MatchDetail(
                    sysml_element=me,
                    req_id=r_elems[best_j].req_id,
                    req_text=r_elems[best_j].text,
                    role=role,
                    score=round(best_score, 4),
                ))
            else:
                matched_model.append(False)
                all_unmatched_model.append(me)

        # Per-req-element: best match
        matched_req = []
        for j, re_ in enumerate(r_elems):
            if not m_elems:
                matched_req.append(False)
                all_unmatched_reqs.append((re_.req_id, role.value, re_.text))
                continue
            best_score = max(score_matrix[i][j] for i in range(len(m_elems)))
            if best_score >= threshold:
                matched_req.append(True)
            else:
                matched_req.append(False)
                all_unmatched_reqs.append((re_.req_id, role.value, re_.text))

        n_mm = sum(matched_model)
        n_rm = sum(matched_req)
        total_model_matched += n_mm
        total_req_matched += n_rm

        mc = n_mm / len(m_elems) if m_elems else (1.0 if not r_elems else 0.0)
        rc = n_rm / len(r_elems) if r_elems else (1.0 if not m_elems else 0.0)
        role_score = (2 * mc * rc / (mc + rc)) if (mc + rc) > 0 else 0.0

        role_breakdown[role.value] = {
            "model_coverage": round(mc, 4),
            "req_coverage":   round(rc, 4),
            "score":          round(role_score, 4),
            "n_model":        len(m_elems),
            "n_req":          len(r_elems),
        }

    # ------------------------------------------------------------------
    # 5. Overall metrics.
    # ------------------------------------------------------------------
    model_coverage = (total_model_matched / total_model) if total_model else 0.0
    req_coverage   = (total_req_matched   / total_req)   if total_req   else 0.0
    if model_coverage + req_coverage > 0:
        semantic_match = (2 * model_coverage * req_coverage
                          / (model_coverage + req_coverage))
    else:
        semantic_match = 0.0

    # Sort matches by score descending for readability
    all_matches.sort(key=lambda m: -m.score)

    return ComparisonReport(
        model_coverage=round(model_coverage, 4),
        req_coverage=round(req_coverage, 4),
        semantic_match=round(semantic_match, 4),
        role_breakdown=role_breakdown,
        matches=all_matches,
        unmatched_model=all_unmatched_model,
        unmatched_reqs=all_unmatched_reqs,
        n_model_elements=total_model,
        n_req_elements=total_req,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAFE_RE = re.compile(r'[^A-Za-z0-9_]')


def _safe(text: str) -> str:
    """Make a string safe for use as a GraphML / Mermaid node identifier."""
    s = _SAFE_RE.sub('_', text)
    if s and s[0].isdigit():
        s = 'n_' + s
    return s or 'n'
