"""
reqgraph.traceability
=====================

Certification-grade **Requirements Verification & Traceability Matrix (RVTM)**.

A semantic-match percentage is not enough to certify a system. A reviewer
(DO-178C, NPR 7123.1, ECSS-E-ST-10) needs, *per requirement*, an auditable
answer to four questions:

1. **Allocation** — which design element(s) satisfy this requirement?
2. **Trace evidence** — is that backed by an explicit model link
   (``satisfy`` / ``derive`` / ``refine`` / ``allocate``), or is it only an
   *inferred* semantic guess that still needs human confirmation?
3. **Verification method** — how will the requirement be verified
   (Test / Analysis / Inspection / Demonstration)?
4. **Quality** — is the requirement itself verifiable (atomic, unambiguous,
   has a modality)?

This module turns a parsed ``SysMLV1Model`` plus a requirement set (and,
optionally, a semantic ``V1ComparisonReport``) into a structured
``TraceabilityMatrix`` with CSV / JSON / GraphML / Markdown exports — the CSV
being the canonical RVTM deliverable an architect hands to a certification
authority.

The key distinction this module makes — and which the raw comparison does not —
is **VERIFIED** (an explicit model trace link exists, auditable) vs
**CANDIDATE** (semantic match only, needs sign-off) vs **GAP** (no allocation).
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from enum import Enum

from .core import Role, _xml_escape
from .corpus import _lexical_similarity
from .quality import enrich
from .sysml_v1_parser import SysMLV1Model, V1Element
from .templates import RUPP_TEMPLATE, Template

# Explicit model relations that constitute auditable traceability evidence,
# in descending strength. ``satisfy`` is the strongest (the design element
# directly satisfies the requirement); ``derive`` links a child requirement to
# its parent (requirement decomposition).
_TRACE_LINKS = ("satisfy", "refine", "allocate", "derive", "trace")


class VerificationMethod(str, Enum):
    """NASA / classic systems-engineering verification methods (IADT)."""
    TEST = "Test"
    ANALYSIS = "Analysis"
    INSPECTION = "Inspection"
    DEMONSTRATION = "Demonstration"


class TraceStatus(str, Enum):
    VERIFIED = "Verified"     # explicit model trace link — auditable
    CANDIDATE = "Candidate"   # semantic match only — needs architect sign-off
    GAP = "Gap"               # no allocation at all — certification finding


@dataclass
class Allocation:
    """One design element allocated to a requirement."""
    element_name: str
    element_type: str
    link_type: str            # satisfy / refine / allocate / derive / trace / inferred
    explicit: bool            # True = from a model relation; False = semantic guess
    confidence: float = 1.0   # 1.0 for explicit links; match score for inferred

    def to_dict(self) -> dict:
        return {"element": self.element_name, "element_type": self.element_type,
                "link_type": self.link_type, "explicit": self.explicit,
                "confidence": round(self.confidence, 4)}


@dataclass
class TraceItem:
    """One row of the RVTM: a requirement and its verification/trace state."""
    req_id: str
    req_text: str
    allocations: list[Allocation]
    trace_status: TraceStatus
    verification_method: VerificationMethod
    verification_status: str          # "Open" until the architect closes it
    quality_pass: bool
    quality_issues: list[str]
    rationale: str

    def to_dict(self) -> dict:
        return {
            "req_id": self.req_id,
            "req_text": self.req_text,
            "allocations": [a.to_dict() for a in self.allocations],
            "allocated_elements": ", ".join(a.element_name for a in self.allocations),
            "trace_status": self.trace_status.value,
            "verification_method": self.verification_method.value,
            "verification_status": self.verification_status,
            "quality_pass": self.quality_pass,
            "quality_issues": self.quality_issues,
            "rationale": self.rationale,
        }


@dataclass
class Finding:
    """An actionable certification finding."""
    finding_id: str
    severity: str       # high / medium / low
    category: str       # traceability / quality / orphan
    subject: str        # requirement id or element name
    issue: str

    def to_dict(self) -> dict:
        return {"id": self.finding_id, "severity": self.severity,
                "category": self.category, "subject": self.subject,
                "issue": self.issue}


@dataclass
class TraceabilityMatrix:
    """The full RVTM plus certification-readiness rollups."""
    items: list[TraceItem]
    orphan_elements: list[dict]      # design elements with no driving requirement
    findings: list[Finding]
    model_name: str = ""

    # ---- readiness rollups ------------------------------------------------
    @property
    def n_requirements(self) -> int:
        return len(self.items)

    @property
    def n_verified(self) -> int:
        return sum(1 for i in self.items if i.trace_status is TraceStatus.VERIFIED)

    @property
    def n_candidate(self) -> int:
        return sum(1 for i in self.items if i.trace_status is TraceStatus.CANDIDATE)

    @property
    def n_gap(self) -> int:
        return sum(1 for i in self.items if i.trace_status is TraceStatus.GAP)

    @property
    def n_quality_pass(self) -> int:
        return sum(1 for i in self.items if i.quality_pass)

    def explicit_trace_rate(self) -> float:
        """Fraction of requirements with an auditable model trace link."""
        return self.n_verified / self.n_requirements if self.items else 0.0

    def trace_completeness(self) -> float:
        """Fraction with *any* allocation (verified or candidate)."""
        n = self.n_requirements
        return (self.n_verified + self.n_candidate) / n if n else 0.0

    def verification_readiness(self) -> float:
        """Fraction that are both explicitly traced *and* quality-passing —
        the requirements that are actually ready to enter verification."""
        n = self.n_requirements
        if not n:
            return 0.0
        ready = sum(1 for i in self.items
                    if i.trace_status is TraceStatus.VERIFIED and i.quality_pass)
        return ready / n

    # ---- exports ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "n_requirements": self.n_requirements,
            "n_verified": self.n_verified,
            "n_candidate": self.n_candidate,
            "n_gap": self.n_gap,
            "n_quality_pass": self.n_quality_pass,
            "explicit_trace_rate": round(self.explicit_trace_rate(), 4),
            "trace_completeness": round(self.trace_completeness(), 4),
            "verification_readiness": round(self.verification_readiness(), 4),
            "items": [i.to_dict() for i in self.items],
            "orphan_elements": self.orphan_elements,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_csv(self) -> str:
        """The canonical RVTM spreadsheet deliverable."""
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["Req ID", "Requirement Text", "Allocated Element(s)",
                    "Link Type", "Trace Status", "Verification Method",
                    "Verification Status", "Quality", "Quality Issues",
                    "Confidence", "Rationale"])
        for it in self.items:
            link_types = ", ".join(sorted({a.link_type for a in it.allocations})) or "-"
            conf = (max((a.confidence for a in it.allocations), default=0.0)
                    if it.allocations else 0.0)
            w.writerow([
                it.req_id, it.req_text,
                "; ".join(a.element_name for a in it.allocations) or "-",
                link_types, it.trace_status.value,
                it.verification_method.value, it.verification_status,
                "PASS" if it.quality_pass else "FAIL",
                "; ".join(it.quality_issues), f"{conf:.2f}", it.rationale,
            ])
        return buf.getvalue()

    def to_markdown(self) -> str:
        lines = [
            f"# Requirements Verification & Traceability Matrix",
            f"",
            f"Model: **{self.model_name or 'model'}** · "
            f"{self.n_requirements} requirements · "
            f"{self.n_verified} verified · {self.n_candidate} candidate · "
            f"{self.n_gap} gap",
            f"",
            f"Explicit trace rate **{self.explicit_trace_rate()*100:.0f}%** · "
            f"Verification readiness **{self.verification_readiness()*100:.0f}%**",
            f"",
            "| Req ID | Requirement | Allocated | Trace | Method | Quality |",
            "|---|---|---|---|---|---|",
        ]
        for it in self.items:
            alloc = "; ".join(a.element_name for a in it.allocations) or "—"
            txt = it.req_text if len(it.req_text) <= 70 else it.req_text[:67] + "…"
            lines.append(
                f"| {it.req_id} | {txt} | {alloc} | {it.trace_status.value} "
                f"| {it.verification_method.value} "
                f"| {'PASS' if it.quality_pass else 'FAIL'} |")
        if self.findings:
            lines += ["", "## Findings", ""]
            for f in self.findings:
                lines.append(f"- **{f.finding_id}** ({f.severity}, {f.category}) "
                             f"— {f.subject}: {f.issue}")
        return "\n".join(lines)

    def to_graphml(self) -> str:
        """Traceability graph: REQUIREMENT and ELEMENT nodes, TRACES edges."""
        out = ['<?xml version="1.0" encoding="UTF-8"?>',
               '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
               '  <key id="node_type" for="node" attr.name="node_type" attr.type="string"/>',
               '  <key id="label"     for="node" attr.name="label"     attr.type="string"/>',
               '  <key id="trace"     for="node" attr.name="trace"     attr.type="string"/>',
               '  <key id="method"    for="node" attr.name="method"    attr.type="string"/>',
               '  <key id="quality"   for="node" attr.name="quality"   attr.type="string"/>',
               '  <key id="link_type" for="edge" attr.name="link_type" attr.type="string"/>',
               '  <key id="explicit"  for="edge" attr.name="explicit"  attr.type="boolean"/>',
               '  <graph edgedefault="directed">']
        elem_ids: dict[str, str] = {}
        for it in self.items:
            rid = f"REQ_{_xml_escape(it.req_id)}"
            out += [f'    <node id="{rid}">',
                    '      <data key="node_type">REQUIREMENT</data>',
                    f'      <data key="label">{_xml_escape(it.req_id)}</data>',
                    f'      <data key="trace">{it.trace_status.value}</data>',
                    f'      <data key="method">{it.verification_method.value}</data>',
                    f'      <data key="quality">{"PASS" if it.quality_pass else "FAIL"}</data>',
                    '    </node>']
            for a in it.allocations:
                if a.element_name not in elem_ids:
                    nid = f"EL_{len(elem_ids)}"
                    elem_ids[a.element_name] = nid
                    out += [f'    <node id="{nid}">',
                            '      <data key="node_type">ELEMENT</data>',
                            f'      <data key="label">{_xml_escape(a.element_name)}</data>',
                            '    </node>']
        edge_i = 0
        for it in self.items:
            rid = f"REQ_{_xml_escape(it.req_id)}"
            for a in it.allocations:
                eid = elem_ids[a.element_name]
                out += [f'    <edge id="t{edge_i}" source="{eid}" target="{rid}">',
                        f'      <data key="link_type">{_xml_escape(a.link_type)}</data>',
                        f'      <data key="explicit">{str(a.explicit).lower()}</data>',
                        '    </edge>']
                edge_i += 1
        out += ['  </graph>', '</graphml>']
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Verification-method assignment (IADT heuristic)
# ---------------------------------------------------------------------------

_QTY_RE = re.compile(
    r"\b\d+(\.\d+)?\s*(%|ms|s|sec|seconds?|m|km|mm|cm|kg|g|v|a|w|hz|khz|mhz|ghz|"
    r"db|dbm|kbps|mbps|fps|rpm|°|deg|m/s|g\b)", re.I)
_COMPARATOR_RE = re.compile(
    r"\b(within|less than|greater than|at least|at most|no more than|"
    r"not exceed|exceed|maximum|minimum|±|<=|>=|<|>)\b", re.I)
_ANALYSIS_RE = re.compile(
    r"\b(margin|factor of safety|reliability|mtbf|availability|cycles?|"
    r"thermal|load|stress|fatigue|budget|worst[- ]case|derived|analysis)\b", re.I)
_DEMO_RE = re.compile(
    r"\b(be able to|ability to|provide .* with the ability|operator shall|"
    r"user shall|pilot shall|demonstrate|usability|operate|interface with the (operator|user|pilot))\b",
    re.I)
_INSPECT_RE = re.compile(
    r"\b(shall include|shall consist of|shall contain|be documented|"
    r"be labell?ed|marking|format|comply with|conform to|conforms? to|"
    r"be made of|material|color|colour|placard)\b", re.I)


def assign_verification_method(text: str) -> VerificationMethod:
    """Suggest a verification method from the requirement wording.

    Heuristic, and meant to be *overridden* by the architect in the RVTM —
    it gives a defensible starting point, not a verdict. Order matters:
    analysis-flavoured wording wins over a bare quantity, etc.
    """
    t = text or ""
    if _ANALYSIS_RE.search(t):
        return VerificationMethod.ANALYSIS
    if _DEMO_RE.search(t):
        return VerificationMethod.DEMONSTRATION
    if _INSPECT_RE.search(t):
        return VerificationMethod.INSPECTION
    if _QTY_RE.search(t) or _COMPARATOR_RE.search(t):
        return VerificationMethod.TEST
    # Functional "shall <verb>" with no measurable quantity → default Test,
    # which the architect can downgrade to Analysis/Inspection as needed.
    return VerificationMethod.TEST


# ---------------------------------------------------------------------------
# Quality gate (NASA "clear, atomic, verifiable" criteria)
# ---------------------------------------------------------------------------

def _quality_gate(graph) -> tuple[bool, list[str]]:
    """Pass/fail a requirement against basic verifiability criteria."""
    q = graph.analysis.get("quality", {})
    issues: list[str] = []
    if q.get("non_atomic"):
        issues.append("not atomic (split into separate requirements)")
    if q.get("compound_requirement"):
        issues.append("compound (multiple modal clauses)")
    if q.get("missing_modality"):
        issues.append("no modality (not an obligation)")
    weak = list(q.get("weak_words", []))
    if weak:
        issues.append("ambiguous: " + ", ".join(weak))
    if q.get("passive_voice"):
        issues.append("passive voice (unclear actor)")
    if q.get("vague_quantifier"):
        issues.append("vague quantifier")
    return (not issues), issues


# ---------------------------------------------------------------------------
# Matrix builder
# ---------------------------------------------------------------------------

def _norm_id(s: str) -> str:
    return re.sub(r"[\s_]+", "", (s or "").strip().upper())


def build_traceability_matrix(
    model: SysMLV1Model,
    items,
    report=None,
    *,
    template: Template = RUPP_TEMPLATE,
    extractor=None,
    roles: tuple = (Role.SUBJECT, Role.PROCESS, Role.OBJECT, Role.CONDITION),
    candidate_threshold: float = 0.5,
    max_candidates: int = 3,
) -> TraceabilityMatrix:
    """Build a Requirements Verification & Traceability Matrix.

    Parameters
    ----------
    model:
        Parsed ``SysMLV1Model`` (carries the explicit ``satisfy``/``derive``…
        relations that constitute auditable traceability evidence).
    items:
        Requirement set: plain strings, ``(id, text)``, or ``(id, text, meta)``.
    report:
        Optional ``V1ComparisonReport`` (from ``compare_v1``). When given, its
        semantic matches supply *candidate* allocations for requirements that
        have no explicit model trace link.
    candidate_threshold:
        Minimum match confidence for a semantic match to be offered as a
        candidate allocation.

    Notes
    -----
    Explicit allocation is established **by requirement ID first** (external
    requirement id == model ``SysML:Requirement`` id), falling back to text
    similarity when no IDs are present. This mirrors how a real RVTM is keyed.
    """
    from .corpus import build_requirement_set_graph

    # 1. parse the requirement set (rules path: cheap, deterministic; ids match
    #    the report's match.req_id because the report uses the same builder)
    rsg = build_requirement_set_graph(
        items, template=template, extractor=extractor,
        roles=roles, threshold=0.0, similarity="lexical")
    for rid in rsg.req_ids:
        enrich(rsg.graphs[rid])

    # 2. index model requirements + explicit trace links
    model_reqs = [e for e in model.elements
                  if e.stereotype.lower() == "requirement"]
    by_id = {_norm_id(e.req_id): e for e in model_reqs if e.req_id}
    elem_by_xid = {e.xmi_id: e for e in model.elements if e.xmi_id}

    # requirement xmi_id -> list[(source design element, link_type)]
    incoming: dict[str, list[tuple]] = {}
    for r in model.relations:
        if r.rel_type in _TRACE_LINKS:
            incoming.setdefault(r.target_id, []).append((r.source_id, r.rel_type))

    # 3. candidate allocations from the semantic report, grouped by req_id
    candidates: dict[str, list] = {}
    if report is not None:
        for m in getattr(report, "matches", []):
            if m.confidence >= candidate_threshold:
                candidates.setdefault(m.req_id, []).append(m)
    for rid in candidates:
        candidates[rid].sort(key=lambda m: m.confidence, reverse=True)

    used_elements: set[str] = set()      # xmi_ids cited as an explicit allocation
    items_out: list[TraceItem] = []
    findings: list[Finding] = []
    fcount = 0

    def _next_finding(sev, cat, subj, issue):
        nonlocal fcount
        fcount += 1
        return Finding(f"{cat[:4].upper()}-{fcount:03d}", sev, cat, subj, issue)

    for rid in rsg.req_ids:
        text = rsg.texts[rid]
        graph = rsg.graphs[rid]
        q_pass, q_issues = _quality_gate(graph)

        # --- find the matching model requirement BY ID (auditable key) ------
        # Exact id match is the only basis for a VERIFIED trace. A split
        # sub-requirement ("...-1") falls back to its parent id; we never bind
        # by fuzzy text for VERIFIED, because a mis-bound "verified" trace is
        # worse than an honest gap.
        key = _norm_id(rid)
        mreq = by_id.get(key)
        if mreq is None:
            base = re.sub(r"-\d+$", "", key)      # strip a split suffix once
            if base != key:
                mreq = by_id.get(base)

        allocations: list[Allocation] = []
        # --- explicit (auditable) allocations -------------------------------
        if mreq is not None:
            for src_id, link in incoming.get(mreq.xmi_id, []):
                src = elem_by_xid.get(src_id)
                if src is None or src.stereotype.lower() == "requirement":
                    continue   # skip requirement-to-requirement derive here
                used_elements.add(src_id)
                allocations.append(Allocation(
                    element_name=src.name,
                    element_type=src.stereotype or src.element_type,
                    link_type=link, explicit=True, confidence=1.0))

        if allocations:
            status = TraceStatus.VERIFIED
            rationale = (f"Auditable: id {rid} matches model requirement "
                         f"{mreq.req_id or mreq.name}, satisfied by "
                         f"{allocations[0].element_name} via "
                         f"{allocations[0].link_type}"
                         + (f" (+{len(allocations)-1} more)"
                            if len(allocations) > 1 else ""))
        else:
            # --- candidate (semantic, needs sign-off) -----------------------
            cands = candidates.get(rid, [])[:max_candidates]
            for m in cands:
                allocations.append(Allocation(
                    element_name=m.element.name,
                    element_type=m.element.stereotype or m.element.element_type,
                    link_type="inferred", explicit=False,
                    confidence=m.confidence))
            if allocations:
                status = TraceStatus.CANDIDATE
                why = ("the requirement id is not in the model"
                       if mreq is None else
                       "the matched model requirement has no satisfy link")
                rationale = (f"No auditable trace ({why}); "
                             f"{len(allocations)} semantic candidate(s), "
                             f"top {allocations[0].confidence:.2f} — needs an "
                             f"explicit satisfy link + sign-off")
            else:
                status = TraceStatus.GAP
                rationale = ("No design element allocated and no semantic "
                             "candidate; requirement is not traceable")

        method = assign_verification_method(text)
        items_out.append(TraceItem(
            req_id=rid, req_text=text, allocations=allocations,
            trace_status=status, verification_method=method,
            verification_status="Open", quality_pass=q_pass,
            quality_issues=q_issues, rationale=rationale))

        # --- findings -------------------------------------------------------
        if status is TraceStatus.GAP:
            findings.append(_next_finding(
                "high", "traceability", rid,
                "no design element satisfies this requirement"))
        elif status is TraceStatus.CANDIDATE:
            findings.append(_next_finding(
                "medium", "traceability", rid,
                "only an inferred (semantic) allocation — add an explicit "
                "satisfy link or confirm"))
        if not q_pass:
            findings.append(_next_finding(
                "medium", "quality", rid,
                "not verifiable as written: " + "; ".join(q_issues)))

    # 4. orphan design elements (no driving requirement) — possible gold-plating
    design_kinds_skip = {"requirement", "package"}
    orphans: list[dict] = []
    semantic_hit = set()
    if report is not None:
        for m in getattr(report, "matches", []):
            semantic_hit.add(m.element.name)
    for e in model.elements:
        if e.stereotype.lower() in design_kinds_skip:
            continue
        if not e.name:
            continue
        if e.xmi_id in used_elements:
            continue
        # a relation source that satisfies *something* counts as allocated
        satisfies_any = any(
            r.source_id == e.xmi_id and r.rel_type in _TRACE_LINKS
            for r in model.relations)
        if satisfies_any or e.name in semantic_hit:
            continue
        # only flag top-level design kinds, not every nested property/port
        if (e.stereotype or e.element_type).lower().split(":")[-1] in (
                "block", "activity", "action", "actiondefinition", "class"):
            orphans.append({"name": e.name,
                            "type": e.stereotype or e.element_type,
                            "package": e.package})
    for o in orphans:
        findings.append(_next_finding(
            "low", "orphan", o["name"],
            "design element has no driving requirement (possible gold-plating)"))

    return TraceabilityMatrix(
        items=items_out, orphan_elements=orphans, findings=findings,
        model_name=model.source_path or "model")
