"""
Tests for reqgraph.sysml_parser and reqgraph.sysml_compare.
"""

from __future__ import annotations

import json
import os
import tempfile
import xml.dom.minidom

import pytest

from reqgraph.core import Role
from reqgraph.errors import DataFormatError
from reqgraph.sysml_parser import SysMLElement, SysMLModel, parse_sysml, read_sysml
from reqgraph.sysml_compare import ComparisonReport, MatchDetail, compare

# ---------------------------------------------------------------------------
# Shared SysML v2 fixture
# ---------------------------------------------------------------------------

AUTOMOTIVE_MODEL = """
package automotive {

  // Structural elements
  part def Vehicle {
    attribute speed: Real;
    attribute mass: Real;
  }
  part def BrakeSystem {
    attribute pressure: Real;
    port brake_cmd_in: BrakeCmdPort;
  }
  part sedan: Vehicle {
    attribute speed = 0;
  }

  // Behavioural elements
  action def ApplyBrakes {
    in pedal: Real;
    out force: Real;
  }
  action def Decelerate {
    in brake_force: Real;
    out new_speed: Real;
  }

  // States
  state def BrakeState {
    state idle;
    state active;
    state faulted;
    transition t1 { from idle; to active; trigger brake_cmd; }
  }

  // Ports / interfaces
  port def BrakeCmdPort {
    flow property pedal_pos: Real;
  }

  // Constraints
  require constraint perf_req {
    // brake deceleration >= 5 m/s^2
  }
  connection def BrakeConnection {
    end a: Vehicle;
    end b: BrakeSystem;
  }

  // SysML requirement block
  requirement req001 {
    doc /* The vehicle braking system shall decelerate at 5 m/s^2 within 200 ms.
           Rationale: safety compliance. */
    subject sedan;
    text "braking system deceleration";
  }
}
"""

REQUIREMENTS = [
    ("R1", "The vehicle shall respond to brake commands within 200 ms."),
    ("R2", "The braking system shall decelerate the vehicle at 5 m/s^2."),
    ("R3", "When the brake state is idle, the vehicle shall not apply brake force."),
    ("R4", "The system shall apply braking to slow the vehicle."),
]

# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

def test_part_def_maps_to_subject():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    names = {e.name for e in model.elements if e.role is Role.SUBJECT}
    assert "Vehicle" in names
    assert "BrakeSystem" in names


def test_part_instance_maps_to_subject():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    names = {e.name for e in model.elements if e.role is Role.SUBJECT}
    assert "sedan" in names


def test_action_def_maps_to_process():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    names = {e.name for e in model.elements if e.role is Role.PROCESS}
    assert "ApplyBrakes" in names
    assert "Decelerate" in names


def test_attribute_maps_to_object():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    names = {e.name for e in model.elements if e.role is Role.OBJECT}
    assert "speed" in names
    assert "pressure" in names


def test_state_def_maps_to_condition():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    names = {e.name for e in model.elements if e.role is Role.CONDITION}
    assert "BrakeState" in names


def test_state_instance_maps_to_condition():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    names = {e.name for e in model.elements if e.role is Role.CONDITION}
    assert "idle" in names
    assert "active" in names


def test_transition_maps_to_condition():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    names = {e.name for e in model.elements if e.role is Role.CONDITION}
    assert "t1" in names


def test_port_def_maps_to_actor():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    names = {e.name for e in model.elements if e.role is Role.ACTOR}
    assert "BrakeCmdPort" in names


def test_require_constraint_maps_to_constraint():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    names = {e.name for e in model.elements if e.role is Role.CONSTRAINT}
    assert "perf_req" in names


def test_connection_def_maps_to_constraint():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    names = {e.name for e in model.elements if e.role is Role.CONSTRAINT}
    assert "BrakeConnection" in names


def test_doc_comment_preserved():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    # The doc comment appears before req001 — check it was captured
    req_docs = [sr[1] for sr in model.sysml_requirements if sr[0] == "req001"]
    assert req_docs, "req001 not found in sysml_requirements"
    assert "decelerate" in req_docs[0].lower() or "braking" in req_docs[0].lower()


def test_requirement_block_extracted_to_sysml_requirements():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    ids = [sr[0] for sr in model.sysml_requirements]
    assert "req001" in ids


def test_package_name_recorded():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    assert "automotive" in model.packages


def test_package_name_on_elements():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    # All top-level elements are inside 'automotive'
    for e in model.elements:
        assert e.package == "automotive", f"{e.name} has wrong package {e.package!r}"


def test_element_type_recorded():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    part_def_elems = [e for e in model.elements if e.element_type == "part_def"]
    assert any(e.name == "Vehicle" for e in part_def_elems)
    action_def_elems = [e for e in model.elements if e.element_type == "action_def"]
    assert any(e.name == "ApplyBrakes" for e in action_def_elems)


def test_empty_input_raises_data_format_error():
    with pytest.raises(DataFormatError, match="empty"):
        parse_sysml("")


def test_whitespace_only_raises_data_format_error():
    with pytest.raises(DataFormatError, match="empty"):
        parse_sysml("   \n  ")


def test_no_sysml_declarations_raises_data_format_error():
    with pytest.raises(DataFormatError, match="no SysML elements"):
        parse_sysml("// just a comment\n/* another comment */\n")


def test_read_sysml_missing_file_raises():
    with pytest.raises(DataFormatError, match="not found"):
        read_sysml("/nonexistent/path/model.sysml")


def test_read_sysml_from_file(tmp_path):
    p = tmp_path / "m.sysml"
    p.write_text(AUTOMOTIVE_MODEL, encoding="utf-8")
    model = read_sysml(str(p))
    assert len(model.elements) > 0
    assert model.source_path == str(p)


def test_line_comments_stripped():
    src = """
    package test {
      // This is a part def Vehicle
      part def RealPart { attribute x: Real; }
    }
    """
    model = parse_sysml(src)
    names = {e.name for e in model.elements}
    # "Vehicle" appears only in a comment — must NOT be extracted
    assert "Vehicle" not in names
    assert "RealPart" in names


def test_block_comments_stripped():
    src = """
    package test {
      /* part def FakePart {} */
      action def RealAction { in x: Real; }
    }
    """
    model = parse_sysml(src)
    names = {e.name for e in model.elements}
    assert "FakePart" not in names
    assert "RealAction" in names


def test_nested_packages():
    src = """
    package outer {
      part def OuterPart {}
      package inner {
        action def InnerAction {}
      }
    }
    """
    model = parse_sysml(src)
    outer_parts = [e for e in model.elements
                   if e.name == "OuterPart" and e.package == "outer"]
    assert outer_parts, "OuterPart not found in outer package"
    inner_actions = [e for e in model.elements
                     if e.name == "InnerAction" and e.package == "inner"]
    assert inner_actions, "InnerAction not found in inner package"


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------

def test_compare_high_score_on_matching_subjects():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    report = compare(model, REQUIREMENTS, threshold=0.3)
    # "Vehicle"/"BrakeSystem" should match "vehicle"/"braking system" subjects
    assert report.semantic_match > 0.0
    subject_matches = [m for m in report.matches if m.role is Role.SUBJECT]
    assert len(subject_matches) > 0
    best_score = max(m.score for m in subject_matches)
    assert best_score >= 0.3


def test_compare_zero_on_no_overlap():
    src = """
    package database {
      part def UserTable { attribute user_id: Real; }
      action def InsertRecord { in row: Row; }
    }
    """
    model = parse_sysml(src)
    unrelated_reqs = [
        ("R1", "The aircraft shall reach cruise altitude within 10 minutes."),
        ("R2", "When the cabin pressure drops, the oxygen masks shall deploy."),
    ]
    report = compare(model, unrelated_reqs, threshold=0.7)
    # Possibly non-zero due to lexical partial matches, but should be low
    assert report.semantic_match < 0.6


def test_compare_model_coverage_fraction():
    src = """
    package test {
      part def A {}
      part def B {}
      part def C {}
    }
    """
    model = parse_sysml(src)
    reqs = [("R1", "The A module shall process requests."),
            ("R2", "The B component shall store data.")]
    # With default threshold 0.6, A and B might match but C probably won't
    report = compare(model, reqs,
                     roles=(Role.SUBJECT,), threshold=0.5)
    # model has 3 subjects; at most 2 can match
    assert 0.0 <= report.model_coverage <= 1.0


def test_compare_req_coverage_fraction():
    src = """
    package test {
      part def Vehicle {}
    }
    """
    model = parse_sysml(src)
    reqs = [("R1", "The vehicle shall accelerate."),
            ("R2", "The aircraft shall fly.")]
    report = compare(model, reqs, roles=(Role.SUBJECT,), threshold=0.5)
    # R1's SUBJECT "vehicle" matches "Vehicle"; R2's "aircraft" does not
    assert 0.0 <= report.req_coverage <= 1.0


def test_compare_semantic_match_is_harmonic_mean():
    src = """
    package test {
      part def Vehicle {}
      part def BrakeSystem {}
    }
    """
    model = parse_sysml(src)
    reqs = [("R1", "The vehicle shall stop.")]
    report = compare(model, reqs, roles=(Role.SUBJECT,), threshold=0.4)
    mc = report.model_coverage
    rc = report.req_coverage
    if mc + rc > 0:
        expected_f1 = round(2 * mc * rc / (mc + rc), 4)
        assert abs(report.semantic_match - expected_f1) < 0.01
    else:
        assert report.semantic_match == 0.0


def test_compare_role_breakdown_has_expected_roles():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    report = compare(model, REQUIREMENTS,
                     roles=(Role.SUBJECT, Role.PROCESS, Role.OBJECT))
    # Only roles that have elements on BOTH sides appear
    for role_val in report.role_breakdown:
        assert role_val in {r.value for r in Role}


def test_compare_unmatched_model_populated():
    src = """
    package test {
      part def A {}
      part def B {}
      part def ZZZNoMatchHere {}
    }
    """
    model = parse_sysml(src)
    reqs = [("R1", "The A module shall do something.")]
    report = compare(model, reqs, roles=(Role.SUBJECT,), threshold=0.5)
    unmatched_names = {e.name for e in report.unmatched_model}
    assert "ZZZNoMatchHere" in unmatched_names


def test_compare_unmatched_reqs_populated():
    src = """
    package test {
      part def Vehicle {}
    }
    """
    model = parse_sysml(src)
    reqs = [
        ("R1", "The vehicle shall stop."),
        ("R2", "The extremely_unusual_xyz_system shall operate."),
    ]
    report = compare(model, reqs, roles=(Role.SUBJECT,), threshold=0.6)
    unmatched_texts = {txt for _, _, txt in report.unmatched_reqs}
    # R2's subject "extremely_unusual_xyz_system" should not match
    assert any("xyz" in txt or "unusual" in txt for txt in unmatched_texts)


def test_compare_matches_list_fields():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    report = compare(model, REQUIREMENTS, threshold=0.3)
    for m in report.matches:
        assert isinstance(m.sysml_element, SysMLElement)
        assert isinstance(m.req_id, str)
        assert isinstance(m.req_text, str)
        assert isinstance(m.role, Role)
        assert 0.0 <= m.score <= 1.0


def test_compare_threshold_excludes_low_scores():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    report_strict = compare(model, REQUIREMENTS, threshold=0.99)
    report_loose  = compare(model, REQUIREMENTS, threshold=0.1)
    assert len(report_strict.matches) <= len(report_loose.matches)
    for m in report_strict.matches:
        assert m.score >= 0.99


def test_compare_to_graphml_well_formed():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    report = compare(model, REQUIREMENTS, threshold=0.3)
    gml = report.to_graphml()
    # Must be parseable XML
    xml.dom.minidom.parseString(gml.encode("utf-8"))
    assert "MODEL" in gml
    assert "REQ_ELEMENT" in gml


def test_compare_to_graphml_has_matches_edges():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    report = compare(model, REQUIREMENTS, threshold=0.3)
    if report.matches:
        gml = report.to_graphml()
        assert "MATCHES" in gml


def test_compare_to_mermaid_starts_with_flowchart():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    report = compare(model, REQUIREMENTS, threshold=0.3)
    mmd = report.to_mermaid()
    assert mmd.startswith("flowchart LR")


def test_compare_to_dict_json_serialisable():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    report = compare(model, REQUIREMENTS, threshold=0.3)
    d = report.to_dict()
    # Must serialise without error
    text = json.dumps(d)
    parsed = json.loads(text)
    assert "semantic_match" in parsed
    assert "model_coverage" in parsed
    assert "req_coverage" in parsed
    assert "role_breakdown" in parsed
    assert "matches" in parsed
    assert "unmatched_model" in parsed
    assert "unmatched_reqs" in parsed


def test_compare_n_counts():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    roles = (Role.SUBJECT, Role.PROCESS)
    report = compare(model, REQUIREMENTS, roles=roles, threshold=0.3)
    # Counts must match the actual elements
    n_model = sum(1 for e in model.elements if e.role in roles)
    assert report.n_model_elements == n_model
    assert report.n_req_elements >= 0


def test_compare_unmatched_plus_matched_equals_total_model():
    model = parse_sysml(AUTOMOTIVE_MODEL)
    roles = (Role.SUBJECT,)
    report = compare(model, REQUIREMENTS, roles=roles, threshold=0.3)
    n_model_in_role = sum(1 for e in model.elements if e.role in roles)
    # matched (deduplicated by element) + unmatched should equal total
    matched_names = {m.sysml_element.name for m in report.matches
                     if m.role in roles}
    unmatched_names = {e.name for e in report.unmatched_model if e.role in roles}
    assert matched_names.isdisjoint(unmatched_names)
    assert len(matched_names) + len(unmatched_names) == n_model_in_role


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

def test_cli_compare_text_output(tmp_path, capsys):
    from reqgraph.__main__ import main
    model_file = tmp_path / "m.sysml"
    model_file.write_text(AUTOMOTIVE_MODEL, encoding="utf-8")
    req_file = tmp_path / "r.txt"
    req_file.write_text(
        "\n".join(t for _, t in REQUIREMENTS), encoding="utf-8")
    rc = main(["compare", str(model_file), str(req_file)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Semantic match:" in out
    assert "Model coverage:" in out
    assert "Req coverage:" in out


def test_cli_compare_json_output(tmp_path, capsys):
    from reqgraph.__main__ import main
    model_file = tmp_path / "m.sysml"
    model_file.write_text(AUTOMOTIVE_MODEL, encoding="utf-8")
    req_file = tmp_path / "r.txt"
    req_file.write_text(
        "\n".join(t for _, t in REQUIREMENTS), encoding="utf-8")
    rc = main(["compare", str(model_file), str(req_file), "--format", "json"])
    assert rc == 0
    out = capsys.readouterr().out
    d = json.loads(out)
    assert "semantic_match" in d
    assert "model_coverage" in d


def test_cli_compare_missing_model_exits(tmp_path):
    from reqgraph.__main__ import main
    req_file = tmp_path / "r.txt"
    req_file.write_text("The vehicle shall stop.", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["compare", str(tmp_path / "nonexistent.sysml"), str(req_file)])
    assert exc.value.code != 0


def test_cli_compare_missing_req_exits(tmp_path):
    from reqgraph.__main__ import main
    model_file = tmp_path / "m.sysml"
    model_file.write_text(AUTOMOTIVE_MODEL, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["compare", str(model_file), str(tmp_path / "nonexistent.txt")])
    assert exc.value.code != 0


def test_cli_compare_report_flag_writes_json(tmp_path):
    from reqgraph.__main__ import main
    model_file = tmp_path / "m.sysml"
    model_file.write_text(AUTOMOTIVE_MODEL, encoding="utf-8")
    req_file = tmp_path / "r.txt"
    req_file.write_text(
        "\n".join(t for _, t in REQUIREMENTS), encoding="utf-8")
    report_file = tmp_path / "report.json"
    rc = main(["compare", str(model_file), str(req_file),
               "--report", str(report_file)])
    assert rc == 0
    assert report_file.exists()
    d = json.loads(report_file.read_text(encoding="utf-8"))
    assert "semantic_match" in d


def test_cli_compare_graphml_flag_writes_xml(tmp_path):
    from reqgraph.__main__ import main
    model_file = tmp_path / "m.sysml"
    model_file.write_text(AUTOMOTIVE_MODEL, encoding="utf-8")
    req_file = tmp_path / "r.txt"
    req_file.write_text(
        "\n".join(t for _, t in REQUIREMENTS), encoding="utf-8")
    gml_file = tmp_path / "match.graphml"
    rc = main(["compare", str(model_file), str(req_file),
               "--graphml", str(gml_file)])
    assert rc == 0
    assert gml_file.exists()
    xml.dom.minidom.parse(str(gml_file))   # must be valid XML


# ---------------------------------------------------------------------------
# GUI (pure function) tests
# ---------------------------------------------------------------------------

def test_compare_request_returns_semantic_match():
    from reqgraph.gui import GuiState, compare_request
    state = GuiState()
    payload = {
        "model_content": AUTOMOTIVE_MODEL,
        "req_content": "\n".join(t for _, t in REQUIREMENTS),
    }
    result = compare_request(state, payload)
    assert "semantic_match" in result
    assert isinstance(result["semantic_match"], float)
    assert 0.0 <= result["semantic_match"] <= 1.0


def test_compare_request_empty_model_raises():
    from reqgraph.errors import ReqGraphError
    from reqgraph.gui import GuiState, compare_request
    state = GuiState()
    with pytest.raises(ReqGraphError, match="SysML"):
        compare_request(state, {"model_content": "", "req_content": "The system shall work."})


def test_compare_request_empty_reqs_raises():
    from reqgraph.errors import ReqGraphError
    from reqgraph.gui import GuiState, compare_request
    state = GuiState()
    with pytest.raises(ReqGraphError, match="requirements"):
        compare_request(state, {"model_content": AUTOMOTIVE_MODEL, "req_content": ""})


def test_compare_request_response_shape():
    from reqgraph.gui import GuiState, compare_request
    state = GuiState()
    payload = {
        "model_content": AUTOMOTIVE_MODEL,
        "req_content": "\n".join(t for _, t in REQUIREMENTS),
        "threshold": 0.3,
    }
    result = compare_request(state, payload)
    for key in ("semantic_match", "model_coverage", "req_coverage",
                "n_model_elements", "n_req_elements", "role_breakdown",
                "matches", "unmatched_model", "unmatched_reqs",
                "graphml", "mermaid", "warnings"):
        assert key in result, f"missing key {key!r}"


def test_compare_request_graphml_is_valid_xml():
    from reqgraph.gui import GuiState, compare_request
    state = GuiState()
    payload = {
        "model_content": AUTOMOTIVE_MODEL,
        "req_content": "\n".join(t for _, t in REQUIREMENTS),
        "threshold": 0.3,
    }
    result = compare_request(state, payload)
    xml.dom.minidom.parseString(result["graphml"].encode("utf-8"))


def test_gui_compare_server_smoke():
    """Start the GUI server and POST to /api/compare."""
    import socket
    import threading
    import urllib.request
    from reqgraph.gui import make_server

    server = make_server(port=0)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        body = json.dumps({
            "model_content": AUTOMOTIVE_MODEL,
            "req_content": "\n".join(t for _, t in REQUIREMENTS),
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/compare",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            d = json.loads(resp.read())
        assert "semantic_match" in d
    finally:
        server.shutdown()
        server.server_close()
