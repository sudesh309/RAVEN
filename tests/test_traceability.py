"""Tests for the certification traceability matrix (RVTM)."""
from __future__ import annotations

import json
import xml.dom.minidom

import pytest

from reqgraph.core import Role
from reqgraph.sysml_v1_parser import V1Element, V1Relation, SysMLV1Model
from reqgraph.sysml_v1_compare import compare_v1
from reqgraph.traceability import (
    VerificationMethod, TraceStatus, build_traceability_matrix,
    assign_verification_method)


# --- a small synthetic model: one auditable satisfy link --------------------

def _model():
    def el(i, name, stereo, role, **kw):
        return V1Element(xmi_id=i, name=name, element_type="uml:Class",
                         stereotype=stereo, role=role, **kw)
    return SysMLV1Model(
        elements=[
            el("brake", "BrakeSystem", "Block", Role.SUBJECT,
               doc="Decelerates the vehicle"),
            el("sensor", "SpeedSensor", "Block", Role.SUBJECT,
               doc="Measures wheel speed continuously"),
            el("req1", "BrakingReq", "Requirement", Role.CONSTRAINT,
               req_id="SYS-001",
               req_text="The brake system shall decelerate the vehicle at 5 m/s2."),
        ],
        relations=[V1Relation("rel1", "brake", "req1", "satisfy")],
        source_path="synthetic",
    )


# id-bearing requirement set exercising all three trace states
_ITEMS = [
    ("SYS-001", "The brake system shall decelerate the vehicle at 5 m/s2 within 200 ms."),
    ("SYS-002", "The speed sensor shall measure wheel speed continuously."),
    ("SYS-003", "The system shall log every detected fault to the maintenance recorder."),
]


def _matrix():
    model = _model()
    report = compare_v1(model, _ITEMS, threshold=0.3, context_hops=2)
    return build_traceability_matrix(model, _ITEMS, report,
                                     candidate_threshold=0.3)


# --- verification-method heuristic ------------------------------------------

def test_method_quantitative_is_test():
    assert assign_verification_method(
        "The system shall respond within 200 ms.") is VerificationMethod.TEST


def test_method_capability_is_demonstration():
    assert assign_verification_method(
        "The operator shall be able to configure the flight plan."
    ) is VerificationMethod.DEMONSTRATION


def test_method_margin_is_analysis():
    assert assign_verification_method(
        "The structure shall maintain a factor of safety of 1.5."
    ) is VerificationMethod.ANALYSIS


def test_method_documentation_is_inspection():
    assert assign_verification_method(
        "The label shall comply with the marking standard."
    ) is VerificationMethod.INSPECTION


# --- trace states -----------------------------------------------------------

def test_explicit_satisfy_link_is_verified():
    m = _matrix()
    sys001 = next(i for i in m.items if i.req_id == "SYS-001")
    assert sys001.trace_status is TraceStatus.VERIFIED
    assert sys001.allocations[0].element_name == "BrakeSystem"
    assert sys001.allocations[0].explicit is True
    assert sys001.allocations[0].link_type == "satisfy"


def test_semantic_only_is_candidate():
    m = _matrix()
    sys002 = next(i for i in m.items if i.req_id == "SYS-002")
    assert sys002.trace_status is TraceStatus.CANDIDATE
    assert sys002.allocations          # has at least one inferred allocation
    assert sys002.allocations[0].explicit is False


def test_no_allocation_is_gap():
    m = _matrix()
    sys003 = next(i for i in m.items if i.req_id == "SYS-003")
    assert sys003.trace_status is TraceStatus.GAP
    assert sys003.allocations == []


def test_verified_requires_exact_id_not_fuzzy_text():
    """A requirement whose id is absent from the model is never Verified,
    even if its text closely matches a model requirement."""
    model = _model()
    items = [("OTHER-99",
              "The brake system shall decelerate the vehicle at 5 m/s2.")]
    m = build_traceability_matrix(model, items)
    assert m.items[0].trace_status is not TraceStatus.VERIFIED


# --- readiness rollups ------------------------------------------------------

def test_explicit_trace_rate_counts_only_verified():
    m = _matrix()
    assert m.n_verified == 1
    assert abs(m.explicit_trace_rate() - 1 / 3) < 1e-6


def test_verification_readiness_requires_trace_and_quality():
    m = _matrix()
    # SYS-001 is verified and clean → ready; readiness counts it
    assert 0.0 < m.verification_readiness() <= 1.0
    assert m.verification_readiness() <= m.explicit_trace_rate() + 1e-9


# --- findings ---------------------------------------------------------------

def test_gap_produces_high_severity_finding():
    m = _matrix()
    gap_finds = [f for f in m.findings
                 if f.category == "traceability" and f.severity == "high"]
    assert any(f.subject == "SYS-003" for f in gap_finds)


def test_candidate_produces_medium_finding():
    m = _matrix()
    assert any(f.severity == "medium" and f.subject == "SYS-002"
               for f in m.findings)


# --- exports ----------------------------------------------------------------

def test_to_csv_has_canonical_rvtm_header():
    m = _matrix()
    first = m.to_csv().splitlines()[0]
    for col in ("Req ID", "Trace Status", "Verification Method", "Quality"):
        assert col in first


def test_to_graphml_well_formed():
    m = _matrix()
    xml.dom.minidom.parseString(m.to_graphml())     # raises if malformed


def test_to_markdown_starts_with_title():
    assert _matrix().to_markdown().startswith("# Requirements")


def test_to_dict_json_serialisable():
    d = _matrix().to_dict()
    json.dumps(d)
    assert d["n_verified"] == 1 and "items" in d and "findings" in d


# --- CLI + GUI integration --------------------------------------------------

def test_cli_compare_v1_writes_rvtm(tmp_path, capsys):
    import reqgraph.__main__ as cli
    model_p = tmp_path / "m.ttl"
    model_p.write_text(
        '@prefix sysmlkg: <http://reqgraph.io/sysml/> .\n'
        '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n'
        '@prefix : <http://x#> .\n'
        ':BrakeSystem a sysmlkg:Block ; rdfs:label "BrakeSystem" ;\n'
        '    sysmlkg:satisfies :R1 .\n'
        ':R1 a sysmlkg:Requirement ; sysmlkg:requirementId "SYS-001" ;\n'
        '    sysmlkg:requirementText "The brake system shall decelerate." .\n',
        encoding="utf-8")
    req_p = tmp_path / "r.csv"
    req_p.write_text("id,text\nSYS-001,The brake system shall decelerate at 5 m/s2.\n",
                     encoding="utf-8")
    out = tmp_path / "rvtm.csv"
    cli.main(["compare-v1", str(model_p), str(req_p),
              "--threshold", "0.3", "--rvtm", str(out)])
    assert out.exists()
    text = out.read_text()
    assert text.startswith("Req ID,")
    assert "SYS-001" in text and "Verified" in text


def test_gui_compare_v1_includes_traceability():
    from reqgraph.gui import GuiState, compare_v1_request
    ttl = ('@prefix sysmlkg: <http://reqgraph.io/sysml/> .\n'
           '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n'
           '@prefix : <http://x#> .\n'
           ':BrakeSystem a sysmlkg:Block ; rdfs:label "BrakeSystem" ;\n'
           '    sysmlkg:satisfies :R1 .\n'
           ':R1 a sysmlkg:Requirement ; sysmlkg:requirementId "SYS-001" ;\n'
           '    sysmlkg:requirementText "The brake system shall decelerate." .\n')
    payload = {"model_content": ttl, "model_format": "turtle",
               "req_content": "id,text\nSYS-001,The brake system shall decelerate at 5 m/s2.\n",
               "req_format": "csv", "threshold": 0.3}
    out = compare_v1_request(GuiState(), payload)
    assert "traceability" in out and "rvtm_csv" in out
    assert out["traceability"]["n_verified"] == 1
    assert out["rvtm_csv"].startswith("Req ID,")
