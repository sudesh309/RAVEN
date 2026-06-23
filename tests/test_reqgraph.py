"""
Test suite for the reqgraph toolkit.

The headline invariant is LOSSLESS ROUND-TRIP: for every backend and template,
graph.generate() must reproduce the input byte-for-byte. Run with:

    python -m pytest tests/ -v
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reqgraph import (RequirementParser, RUPP_TEMPLATE, EARS_TEMPLATE, Role,
                      Template, register_template, build_requirement)
from reqgraph.extractors import RuleExtractor, SpacyExtractor

REQS = [
    "The flight management system shall calculate the optimal cruise altitude.",
    "When the cabin altitude exceeds 14,000 feet, the oxygen system shall deploy "
    "the passenger oxygen masks within 4 seconds.",
    "While the aircraft is on the ground, the avionics suite shall provide the pilot "
    "with the ability to configure the flight plan if the parking brake is set.",
    "As soon as an engine fire is detected, the engine control unit shall shut off "
    "the affected engine and activate the fire suppression system within 500 milliseconds.",
    "The system shall not transmit the message during radio silence.",
    "Where dual redundancy is installed, the system should report the channel status.",
]


# --- lossless round-trip (the core guarantee) ------------------------------

@pytest.mark.parametrize("text", REQS)
def test_roundtrip_rules(text):
    g = RequirementParser(RUPP_TEMPLATE, RuleExtractor()).parse(text)
    assert g.generate() == text


@pytest.mark.parametrize("text", REQS)
def test_roundtrip_spacy(text):
    ex = SpacyExtractor()
    if not ex.available():
        pytest.skip("spaCy model not installed")
    g = RequirementParser(RUPP_TEMPLATE, ex).parse(text)
    assert g.generate() == text


def test_roundtrip_survives_garbage():
    """Even on text the parser cannot classify, tiling stays lossless."""
    weird = "   xyzzy ?? 42 ... !!  "
    g = RequirementParser(RUPP_TEMPLATE).parse(weird)
    assert g.generate() == weird


# --- semantic structure ----------------------------------------------------

def test_elements_extracted():
    g = RequirementParser(RUPP_TEMPLATE).parse(REQS[1])
    roles = {n.role for n in g.elements()}
    assert Role.CONDITION in roles
    assert Role.SUBJECT in roles
    assert Role.MODALITY in roles
    assert Role.CONSTRAINT in roles


def test_compound_action_operator():
    from reqgraph.core import Role as R
    g = RequirementParser(RUPP_TEMPLATE).parse(REQS[3])
    ops = [n for n in g.nodes.values() if n.role is R.OPERATOR]
    assert ops and ops[0].attrs.get("operator") == "AND"
    assert len(g.by_role(R.PROCESS)) == 2


def test_modality_obligation():
    g = RequirementParser(RUPP_TEMPLATE).parse(REQS[0])
    modal = g.by_role(Role.MODALITY)[0]
    assert "mandatory" in modal.attrs["obligation"]


# --- multi-requirement splitting --------------------------------------------

def test_split_two_shall_conjunction():
    from reqgraph import split_requirements
    text = "The system shall open the valve and the controller shall log the event."
    segs = split_requirements(text)
    assert segs == ["The system shall open the valve",
                    "the controller shall log the event."]


def test_split_two_sentences():
    from reqgraph import split_requirements
    text = "The system shall open the valve. The system shall log the event."
    assert split_requirements(text) == [
        "The system shall open the valve.", "The system shall log the event."]


def test_split_comma_splice_and_semicolon():
    from reqgraph import split_requirements
    assert len(split_requirements(
        "The system shall open the valve, the operator shall confirm the action.")) == 2
    assert len(split_requirements(
        "The system shall open the valve; the system shall log the event.")) == 2


def test_split_keeps_compound_action_atomic():
    from reqgraph import split_requirements
    # one modality governing two actions stays ONE requirement
    assert split_requirements(REQS[3]) == [REQS[3]]
    # "shall not" is a single modality token, not two
    assert split_requirements(REQS[4]) == [REQS[4]]
    # embedded second modality without its own clause does not split
    text = "The system shall ensure the operator can access the logs."
    assert split_requirements(text) == [text]


def test_split_ignores_decimals_and_object_lists():
    from reqgraph import split_requirements
    text = ("The pump shall start within 4.5 seconds and "
            "the controller shall record the time.")
    assert split_requirements(text) == [
        "The pump shall start within 4.5 seconds",
        "the controller shall record the time."]
    text = "The system shall monitor the pump and the valve and the controller shall log events."
    assert split_requirements(text) == [
        "The system shall monitor the pump and the valve",
        "the controller shall log events."]


def test_parse_many_roundtrips_and_ids():
    p = RequirementParser(RUPP_TEMPLATE)
    text = "The system shall open the valve and the controller shall log the event."
    graphs = p.parse_many(text, metadata={"id": "REQ-9"})
    assert len(graphs) == 2
    for g, seg in zip(graphs, p.split(text)):
        assert g.generate() == seg
    assert [g.metadata["id"] for g in graphs] == ["REQ-9-1", "REQ-9-2"]
    subjects = [g.by_role(Role.SUBJECT)[0].text.strip() for g in graphs]
    assert subjects == ["The system", "the controller"]


# --- templates -------------------------------------------------------------

def test_ears_template():
    text = ("If the airspeed drops below the stall speed, then the stall warning "
            "system shall activate the stick shaker.")
    g = RequirementParser(EARS_TEMPLATE).parse(text)
    assert g.generate() == text
    assert g.by_role(Role.CONDITION)


def test_custom_template():
    contract = Template(
        name="CONTRACT",
        modality_keywords=("is required to", "is permitted to"),
        user_interaction_open=(), interface_markers=(),
    )
    register_template(contract)
    text = "The ground station is required to acknowledge the telemetry packet within 2 seconds."
    g = RequirementParser(contract).parse(text)
    assert g.generate() == text
    assert g.by_role(Role.MODALITY)[0].text.strip() == "is required to"


def test_build_requirement_roundtrips():
    g = build_requirement({
        Role.CONDITION: "When the landing gear is selected down",
        Role.SUBJECT: "the landing gear control system",
        Role.MODALITY: "shall",
        Role.PROCESS: "extend",
        Role.OBJECT: "the main landing gear",
        Role.CONSTRAINT: "within 10 seconds",
    })
    out = g.generate()
    assert out.startswith("When the landing gear is selected down,")
    assert RequirementParser(RUPP_TEMPLATE).parse(out).generate() == out


# --- serialisation ---------------------------------------------------------

def test_json_roundtrip():
    from reqgraph import RequirementGraph
    g = RequirementParser(RUPP_TEMPLATE).parse(REQS[3])
    reloaded = RequirementGraph.from_dict(g.to_dict())
    assert reloaded.generate() == REQS[3]


def test_exporters_smoke():
    g = RequirementParser(RUPP_TEMPLATE).parse(REQS[1])
    assert "flowchart" in g.to_mermaid()
    assert "digraph" in g.to_dot()
    assert "CREATE" in g.to_cypher()


def test_graphml_is_well_formed_xml():
    import xml.dom.minidom as minidom
    g = RequirementParser(RUPP_TEMPLATE).parse(REQS[1])
    gml = g.to_graphml()
    assert gml.startswith("<?xml")
    assert "<graphml" in gml and "edgedefault=\"directed\"" in gml
    # must parse as valid XML (raises otherwise)
    minidom.parseString(gml)
    # nodes carry their role; condition span is present
    assert ">CONDITION<" in gml
    assert ">SUBJECT<" in gml


def test_graphml_escapes_special_chars():
    import xml.dom.minidom as minidom
    text = 'The system shall display the "ALT < HOLD" & status annunciation.'
    g = RequirementParser(RUPP_TEMPLATE).parse(text)
    gml = g.to_graphml()
    minidom.parseString(gml)             # raw <, &, " would break this
    assert g.generate() == text          # still lossless


def test_turtle_export_structure():
    g = RequirementParser(RUPP_TEMPLATE).parse(REQS[1])
    ttl = g.to_turtle()
    assert "@prefix rg:" in ttl
    assert "a rg:SUBJECT" in ttl
    assert "a rg:MODALITY" in ttl
    # relationship predicates between resources
    assert "rg:HAS_MODALITY" in ttl
    # text is escaped/quoted; quotes in content must not break the literal
    text = 'The system shall display the "ALT HOLD" annunciation.'
    ttl2 = RequirementParser(RUPP_TEMPLATE).parse(text).to_turtle()
    assert '\\"ALT HOLD\\"' in ttl2


def test_graphml_turtle_skip_glue_by_default():
    g = RequirementParser(RUPP_TEMPLATE).parse(REQS[1])
    assert ">GLUE<" not in g.to_graphml()
    assert "rg:GLUE" not in g.to_turtle()
    assert ">GLUE<" in g.to_graphml(show_glue=True)


def test_cli_parse_graphml_and_turtle(capsys):
    from reqgraph.__main__ import main
    assert main(["parse", REQS[0], "--format", "graphml"]) == 0
    assert "<graphml" in capsys.readouterr().out
    assert main(["parse", REQS[0], "--format", "turtle"]) == 0
    assert "@prefix rg:" in capsys.readouterr().out


# --- quality / classification ----------------------------------------------

def test_quality_and_classification():
    from reqgraph.quality import enrich
    g = enrich(RequirementParser(RUPP_TEMPLATE).parse(
        "The system should be fast and user-friendly and reliable."))
    q = g.analysis["quality"]
    assert "fast" in q["weak_words"]
    assert q["non_atomic"] is True
    assert g.analysis["type"] in {"functional", "performance", "interface",
                                  "safety", "usability"}


def test_compound_requirement_smell():
    from reqgraph.quality import check_quality
    # two independent modality clauses → compound smell fires
    q = check_quality(
        "The system shall open the valve and the controller shall log the event.")
    assert q["compound_requirement"] is True
    # single modality with compound action → NOT compound
    q2 = check_quality(
        "As soon as a fire is detected, the system shall shut off the engine "
        "and activate the suppression system within 500 milliseconds.")
    assert q2["compound_requirement"] is False
    # "shall not" counts as one modality
    assert check_quality("The system shall not transmit.")["compound_requirement"] is False


# --- batch I/O -------------------------------------------------------------

def test_reqif_roundtrip():
    from reqgraph.io_formats import write_reqif, read_reqif
    items = [(f"REQ-{i+1:03d}", t) for i, t in enumerate(REQS)]
    with tempfile.TemporaryDirectory() as d:
        path = write_reqif(items, os.path.join(d, "r.reqif"))
        back = read_reqif(path)
    assert [t for _, t, *_ in back] == [t for _, t in items]


def test_csv_roundtrip():
    from reqgraph.io_formats import write_csv, read_requirements_csv
    items = [(f"REQ-{i+1:03d}", t) for i, t in enumerate(REQS)]
    with tempfile.TemporaryDirectory() as d:
        path = write_csv(items, os.path.join(d, "r.csv"))
        back = read_requirements_csv(path)
    assert [t for _, t, *_ in back] == [t for _, t in items]


# --- requirement set: cross-requirement SUBJECT/OBJECT connections ---------

SET_REQS = [
    ("R1", "The flight management system shall calculate the optimal cruise altitude."),
    ("R2", "The flight management system shall log every altitude change to the "
           "maintenance recorder."),
    ("R3", "The pilot shall be able to override the calculated cruise altitude."),
    ("R4", "The oxygen system shall deploy the passenger oxygen masks when the "
           "cabin altitude exceeds 14,000 feet."),
]


def test_requirement_set_finds_subject_and_object_connections():
    from reqgraph.corpus import build_requirement_set_graph
    rsg = build_requirement_set_graph(SET_REQS)
    assert rsg.req_ids == ["R1", "R2", "R3", "R4"]
    pairs = {(c.a.req_id, c.b.req_id, c.role.value) for c in rsg.connections}
    # R1 and R2 share the exact same SUBJECT
    assert ("R1", "R2", "SUBJECT") in pairs
    # R1 and R3 share a near-identical OBJECT (optimal vs calculated cruise altitude)
    assert ("R1", "R3", "OBJECT") in pairs
    # unrelated subjects ("flight management system" vs "oxygen system") must
    # not connect just because both spans contain a determiner + "system"
    assert not any(c.role is Role.SUBJECT and {c.a.req_id, c.b.req_id} == {"R1", "R4"}
                   for c in rsg.connections)


def test_requirement_set_respects_threshold():
    from reqgraph.corpus import build_requirement_set_graph
    rsg = build_requirement_set_graph(SET_REQS, threshold=0.999)
    # only the byte-identical SUBJECT survives an (almost) exact-match threshold
    assert all(c.score >= 0.999 for c in rsg.connections)
    assert ("R1", "R3") not in {(c.a.req_id, c.b.req_id) for c in rsg.connections}


def test_requirement_set_splits_compound_items():
    from reqgraph.corpus import build_requirement_set_graph
    items = ["The system shall open the valve and the controller shall log the event."]
    rsg = build_requirement_set_graph(items)
    assert rsg.req_ids == ["REQ-1-1", "REQ-1-2"]
    assert rsg.graphs["REQ-1-1"].generate() == "The system shall open the valve"
    assert rsg.graphs["REQ-1-2"].generate() == "the controller shall log the event."


def test_requirement_set_exporters():
    import xml.dom.minidom as minidom
    from reqgraph.corpus import build_requirement_set_graph
    rsg = build_requirement_set_graph(SET_REQS)

    mer = rsg.to_mermaid()
    assert mer.startswith("flowchart")
    assert "R1" in mer and "-->" in mer

    dot = rsg.to_dot()
    assert dot.startswith("digraph RequirementSet")

    gml = rsg.to_graphml()
    minidom.parseString(gml)        # must be well-formed XML
    assert "<graphml" in gml

    ttl = rsg.to_turtle()
    assert "@prefix rg:" in ttl
    assert "a rg:Requirement" in ttl

    cyp = rsg.to_cypher()
    assert "CREATE (:Requirement" in cyp
    assert "SIMILAR_SUBJECT" in cyp or "SIMILAR_OBJECT" in cyp

    d = rsg.to_dict()
    assert len(d["requirements"]) == 4
    assert all({"req_a", "req_b", "role", "score"} <= c.keys() for c in d["connections"])


def test_cli_connections_text_and_json(tmp_path, capsys):
    pytest.importorskip("pandas")
    from reqgraph.__main__ import main
    csv_in = tmp_path / "set.csv"
    csv_in.write_text(
        "id,text\n"
        "R1,The flight management system shall calculate the optimal cruise altitude.\n"
        "R2,The flight management system shall log every altitude change to the "
        "maintenance recorder.\n",
        encoding="utf-8")
    assert main(["connections", str(csv_in)]) == 0
    out = capsys.readouterr().out
    assert "SUBJECT" in out

    assert main(["connections", str(csv_in), "--format", "json"]) == 0
    payload = capsys.readouterr().out
    assert '"connections"' in payload and '"requirements"' in payload


# --- BERT tagger (optional, slow) ------------------------------------------

def test_bert_tagger_trains_and_roundtrips():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from reqgraph.nlp import BertTokenTagger
    from reqgraph.extractors import BertTaggerExtractor

    def span(text, items):
        return text, [(text.index(s), text.index(s) + len(s), r) for s, r in items]

    data = [
        span("The sensor shall measure the cabin pressure.",
             [("The sensor", Role.SUBJECT), ("shall", Role.MODALITY),
              ("measure", Role.PROCESS), ("the cabin pressure", Role.OBJECT)]),
        span("The autopilot shall maintain the heading.",
             [("The autopilot", Role.SUBJECT), ("shall", Role.MODALITY),
              ("maintain", Role.PROCESS), ("the heading", Role.OBJECT)]),
    ]
    tagger = BertTokenTagger().train(data, epochs=5, verbose=False)
    text = "The system shall measure the speed."
    g = RequirementParser(RUPP_TEMPLATE, BertTaggerExtractor(tagger=tagger)).parse(text)
    assert g.generate() == text  # lossless regardless of tag accuracy


# --- seed corpus + CLI -----------------------------------------------------

def test_seed_data_valid():
    from reqgraph.seed_data import training_examples, validate
    assert validate() == 30
    for text, spans in training_examples():
        for (s, e, role) in spans:
            assert text[s:e].strip()
            assert isinstance(role, Role)


def test_cli_parse_smoke(capsys):
    from reqgraph.__main__ import main
    rc = main(["parse", "The system shall close the valve.", "--format", "elements"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PROCESS" in out and "close" in out


def test_cli_batch_and_analyze(tmp_path, capsys):
    from reqgraph.__main__ import main
    csv_in = tmp_path / "in.csv"
    csv_in.write_text("id,text\nR1,The autopilot shall maintain the heading.\n", encoding="utf-8")
    rc = main(["batch", str(csv_in), "--out", str(tmp_path / "out.reqif")])
    assert rc == 0
    assert (tmp_path / "out.reqif").exists()


# --- production hardening ----------------------------------------------------

def test_graph_verify_passes_and_detects_corruption():
    from reqgraph import GraphIntegrityError
    g = RequirementParser(RUPP_TEMPLATE).parse(REQS[0])
    assert g.verify() is True
    g.leaf_order.append("ghost_node")
    with pytest.raises(GraphIntegrityError):
        g.verify()


def test_add_edge_rejects_unknown_nodes():
    from reqgraph import RequirementGraph, GraphIntegrityError, Rel
    g = RequirementGraph()
    n = g.add_node(Role.SUBJECT, "x")
    with pytest.raises(GraphIntegrityError):
        g.add_edge(n.id, "nope", Rel.NEXT)


def test_empty_and_invalid_input():
    p = RequirementParser(RUPP_TEMPLATE)
    assert p.parse("").generate() == ""
    with pytest.raises(TypeError):
        p.parse(None)


def test_tiler_survives_bad_claims():
    from reqgraph import tile_to_graph
    text = "The system shall work."
    bad_claims = [(-5, 3, Role.SUBJECT, {}),        # clamped
                  (10, 9999, Role.OBJECT, {}),      # clamped
                  (4, 4, Role.PROCESS, {}),         # empty -> dropped
                  ("x", "y", Role.PROCESS, {})]     # malformed -> dropped
    g = tile_to_graph(text, bad_claims)
    assert g.generate() == text
    assert g.verify()


def test_mermaid_dot_escape_quotes():
    text = 'The system shall display the "ALT HOLD" annunciation.'
    g = RequirementParser(RUPP_TEMPLATE).parse(text)
    assert g.generate() == text
    mer = g.to_mermaid()
    # node labels must not contain raw double quotes inside quoted labels
    for line in mer.splitlines():
        if '["' in line:
            inner = line.split('["', 1)[1].rsplit('"', 1)[0]
            assert '"' not in inner
    assert '\\"' in g.to_dot() or '"' not in g.to_dot().split('label="')[1].split('"];')[0]


def test_template_frozen_and_pattern_cache():
    import dataclasses
    from reqgraph.extractors import _patterns
    with pytest.raises(dataclasses.FrozenInstanceError):
        RUPP_TEMPLATE.name = "mutated"
    assert _patterns(RUPP_TEMPLATE) is _patterns(RUPP_TEMPLATE)  # memoised


def test_dataframe_missing_column_raises():
    from reqgraph import DataFormatError
    from reqgraph.io_formats import read_requirements_csv
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bad.csv")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("wrong_column\nfoo\n")
        with pytest.raises(DataFormatError):
            read_requirements_csv(path)


def test_from_dict_rejects_garbage():
    from reqgraph import RequirementGraph, DataFormatError
    with pytest.raises(DataFormatError):
        RequirementGraph.from_dict({"nodes": [{"id": "a"}], "edges": [], "leaf_order": []})


# --- GUI ---------------------------------------------------------------------

def test_gui_parse_request_payload():
    from reqgraph.gui import GuiState, parse_request
    d = parse_request(GuiState(), {"text": REQS[1], "backend": "rules"})
    assert d["kpis"]["roundtrip_ok"] is True
    assert d["kpis"]["n_elements"] >= 5
    assert 0 < d["kpis"]["coverage_pct"] <= 100
    assert d["tree"]["role"] == "ROOT" and d["tree"]["children"]
    assert "".join(t["text"] for t in d["tiles"]) == REQS[1]   # tiling = lossless
    roles = {e["role"] for e in d["elements"]}
    assert {"CONDITION", "SUBJECT", "MODALITY", "CONSTRAINT"} <= roles
    # knowledge-graph exports are included in the payload
    assert "<graphml" in d["graphml"]
    assert "@prefix rg:" in d["turtle"]
    assert "CREATE" in d["cypher"]


def test_gui_parse_request_rejects_empty():
    from reqgraph import ReqGraphError
    from reqgraph.gui import GuiState, parse_request
    with pytest.raises(ReqGraphError):
        parse_request(GuiState(), {"text": "   "})


def test_gui_connections_request_payload():
    from reqgraph.gui import GuiState, connections_request
    texts = [t for _, t in SET_REQS]
    d = connections_request(GuiState(), {"texts": texts})
    assert d["n_requirements"] == 4
    assert d["n_connections"] >= 2
    assert {c["role"] for c in d["connections"]} <= {"SUBJECT", "OBJECT"}
    assert "flowchart" in d["mermaid"]
    assert "<graphml" in d["graphml"]
    assert "@prefix rg:" in d["turtle"]
    assert "CREATE" in d["cypher"]


def test_gui_connections_request_rejects_empty():
    from reqgraph import ReqGraphError
    from reqgraph.gui import GuiState, connections_request
    with pytest.raises(ReqGraphError):
        connections_request(GuiState(), {"texts": ["   ", ""]})


def test_gui_server_smoke():
    import json as _json
    import threading
    import urllib.request
    from reqgraph.gui import make_server

    server = make_server(port=0)            # free port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(f"{base}/", timeout=10).read().decode()
        assert "reqgraph studio" in page
        info = _json.load(urllib.request.urlopen(f"{base}/api/info", timeout=10))
        assert info["backends"]["rules"] is True
        # config explainer content is served for the UI
        assert "how" in info["backend_info"]["rules"]
        assert info["template_info"]["IREB-Rupp"]
        req = urllib.request.Request(
            f"{base}/api/parse",
            data=_json.dumps({"text": REQS[0]}).encode(),
            headers={"Content-Type": "application/json"})
        out = _json.load(urllib.request.urlopen(req, timeout=10))
        assert out["kpis"]["roundtrip_ok"] is True
        creq = urllib.request.Request(
            f"{base}/api/connections",
            data=_json.dumps({"texts": [t for _, t in SET_REQS]}).encode(),
            headers={"Content-Type": "application/json"})
        cout = _json.load(urllib.request.urlopen(creq, timeout=10))
        assert cout["n_requirements"] == 4
        # /api/export with CSV text content (use csv module for proper quoting)
        import csv as _csv, io as _io
        buf = _io.StringIO()
        w = _csv.writer(buf)
        w.writerow(["id", "text"])
        for rid, text in SET_REQS:
            w.writerow([rid, text])
        csv_content = buf.getvalue()
        ereq = urllib.request.Request(
            f"{base}/api/export",
            data=_json.dumps({"format": "csv", "content": csv_content}).encode(),
            headers={"Content-Type": "application/json"})
        eout = _json.load(urllib.request.urlopen(ereq, timeout=10))
        assert eout["n_requirements"] == 4
        assert "<graphml" in eout["graphml"]
        assert "csv_data" in eout
    finally:
        server.shutdown()
        server.server_close()


# --- JSON reader + extended attributes -------------------------------------------

def test_read_requirements_json_array_of_objects(tmp_path):
    import json as _json
    from reqgraph.io_formats import read_requirements_json
    data = [
        {"id": "R1", "text": "The system shall log errors.", "rationale": "audit trail"},
        {"id": "R2", "text": "The sensor shall measure pressure.", "applicability": "all"},
    ]
    p = tmp_path / "reqs.json"
    p.write_text(_json.dumps(data), encoding="utf-8")
    items = read_requirements_json(str(p))
    assert len(items) == 2
    rid, text, meta = items[0]
    assert rid == "R1"
    assert "log errors" in text
    assert meta["rationale"] == "audit trail"
    rid2, text2, meta2 = items[1]
    assert meta2["applicability"] == "all"


def test_read_requirements_json_dict_keyed(tmp_path):
    import json as _json
    from reqgraph.io_formats import read_requirements_json
    data = {"R1": {"text": "The valve shall close within 2 seconds.", "applicability": "type A"},
            "R2": {"text": "The pump shall maintain pressure."}}
    p = tmp_path / "d.json"
    p.write_text(_json.dumps(data), encoding="utf-8")
    items = read_requirements_json(str(p))
    by_id = {rid: (text, meta) for rid, text, meta in items}
    assert "valve" in by_id["R1"][0]
    assert by_id["R1"][1]["applicability"] == "type A"   # key normalised, value preserved


def test_read_requirements_json_plain_string_dict(tmp_path):
    import json as _json
    from reqgraph.io_formats import read_requirements_json
    data = {"R1": "The system shall respond.", "R2": "The sensor shall measure."}
    p = tmp_path / "plain.json"
    p.write_text(_json.dumps(data), encoding="utf-8")
    items = read_requirements_json(str(p))
    assert len(items) == 2
    rid, text, meta = items[0]
    assert rid == "R1"
    assert meta == {}


def test_read_requirements_json_bad_shape_raises(tmp_path):
    from reqgraph import DataFormatError
    from reqgraph.io_formats import read_requirements_json
    p = tmp_path / "bad.json"
    p.write_text('"just a string"', encoding="utf-8")
    with pytest.raises(DataFormatError):
        read_requirements_json(str(p))


def test_rows_from_df_captures_extra_columns():
    import pandas as pd
    from reqgraph.io_formats import _rows_from_df
    df = pd.DataFrame([
        {"id": "R1", "text": "The system shall log.", "rationale": "audit", "applicability": "all"},
    ])
    items = _rows_from_df(df, "text", "id")
    rid, text, meta = items[0]
    assert rid == "R1"
    assert meta["rationale"] == "audit"
    assert meta["applicability"] == "all"


def test_rows_from_df_normalises_column_names():
    import pandas as pd
    from reqgraph.io_formats import _rows_from_df
    df = pd.DataFrame([{"id": "R1", "text": "x", "Additional Info": "note", "Req-Source": "sys"}])
    items = _rows_from_df(df, "text", "id")
    _, _, meta = items[0]
    assert "additional_info" in meta
    assert "req_source" in meta


def test_read_requirements_csv_with_extra_columns(tmp_path):
    from reqgraph.io_formats import read_requirements_csv
    p = tmp_path / "r.csv"
    p.write_text("id,text,rationale,applicability\n"
                 "R1,The system shall log.,audit trail,always\n", encoding="utf-8")
    items = read_requirements_csv(str(p))
    rid, text, meta = items[0]
    assert meta["rationale"] == "audit trail"
    assert meta["applicability"] == "always"


def test_requirements_to_dataframe_includes_metadata_columns():
    from reqgraph.io_formats import requirements_to_dataframe
    items = [
        ("R1", "The system shall log errors.", {"rationale": "audit", "applicability": "all"}),
        ("R2", "The sensor shall measure pressure.", {"rationale": "safety"}),
    ]
    df = requirements_to_dataframe(items)
    cols = list(df.columns)
    assert "rationale" in cols
    assert "applicability" in cols
    # metadata columns must appear before type column
    assert cols.index("rationale") < cols.index("type")
    assert df.iloc[0]["rationale"] == "audit"
    assert df.iloc[1]["applicability"] == ""   # missing → empty string


def test_requirement_set_graph_stores_metadata():
    from reqgraph.corpus import build_requirement_set_graph
    items = [
        ("R1", "The flight management system shall calculate the optimal cruise altitude.",
         {"rationale": "fuel efficiency"}),
        ("R2", "The flight management system shall log every altitude change.",
         {"applicability": "all aircraft"}),
    ]
    rsg = build_requirement_set_graph(items)
    assert rsg.metadata["R1"]["rationale"] == "fuel efficiency"
    assert rsg.metadata["R2"]["applicability"] == "all aircraft"


def test_requirement_set_graph_to_dict_includes_metadata():
    from reqgraph.corpus import build_requirement_set_graph
    items = [("R1", "The system shall log.", {"rationale": "because"})]
    rsg = build_requirement_set_graph(items)
    d = rsg.to_dict()
    req_entry = d["requirements"][0]
    assert req_entry["rationale"] == "because"


def test_to_element_graphml_is_well_formed_xml():
    import xml.dom.minidom
    from reqgraph.corpus import build_requirement_set_graph
    rsg = build_requirement_set_graph(SET_REQS)
    gml = rsg.to_element_graphml()
    xml.dom.minidom.parseString(gml.encode("utf-8"))   # raises if not well-formed


def test_to_element_graphml_has_req_and_element_nodes():
    import xml.etree.ElementTree as ET
    from reqgraph.corpus import build_requirement_set_graph
    rsg = build_requirement_set_graph(SET_REQS)
    gml = rsg.to_element_graphml()
    root = ET.fromstring(gml)
    ns = "http://graphml.graphdrawing.org/xmlns"
    nodes = root.findall(f".//{{{ns}}}node")
    node_ids = {n.get("id") for n in nodes}
    # all requirement ids must appear as REQ nodes
    for rid in rsg.req_ids:
        assert rid in node_ids
    # at least one ELEMENT node must exist (double-underscore namespaced)
    elem_nodes = [n for n in node_ids if "__" in n]
    assert len(elem_nodes) > 0


def test_to_element_graphml_has_similarity_edges():
    import xml.etree.ElementTree as ET
    from reqgraph.corpus import build_requirement_set_graph
    rsg = build_requirement_set_graph(SET_REQS, threshold=0.5)
    gml = rsg.to_element_graphml()
    assert "SIMILAR_" in gml   # at least one cross-req similarity edge


def test_to_element_graphml_has_has_element_edges():
    from reqgraph.corpus import build_requirement_set_graph
    rsg = build_requirement_set_graph(SET_REQS)
    gml = rsg.to_element_graphml()
    assert "HAS_ELEMENT" in gml


def test_to_element_graphml_metadata_in_req_node():
    from reqgraph.corpus import build_requirement_set_graph
    items = [
        ("R1", "The flight management system shall calculate the optimal cruise altitude.",
         {"rationale": "fuel efficiency", "applicability": "all aircraft"}),
        ("R2", "The flight management system shall log every altitude change.",
         {}),
    ]
    rsg = build_requirement_set_graph(items)
    gml = rsg.to_element_graphml()
    assert "fuel efficiency" in gml   # metadata JSON appears in graphml


# --- CLI export subcommand ---------------------------------------------------

def test_cli_export_csv_json_graphml(tmp_path, capsys):
    import xml.dom.minidom
    pandas = pytest.importorskip("pandas")
    from reqgraph.__main__ import main
    csv_in = tmp_path / "in.csv"
    import csv as _csv, io as _io
    buf = _io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["id", "text", "rationale"])
    for rid, text in SET_REQS:
        w.writerow([rid, text, f"rationale for {rid}"])
    csv_in.write_text(buf.getvalue(), encoding="utf-8")
    prefix = str(tmp_path / "out")
    rc = main(["export", str(csv_in), "--out-prefix", prefix])
    assert rc == 0
    assert (tmp_path / "out.csv").exists()
    assert (tmp_path / "out.json").exists()
    assert (tmp_path / "out.graphml").exists()
    df = pandas.read_csv(str(tmp_path / "out.csv"))
    assert len(df) == len(SET_REQS)
    assert "rationale" in df.columns
    import json as _json
    rows = _json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert len(rows) == len(SET_REQS)
    xml.dom.minidom.parse(str(tmp_path / "out.graphml"))


def test_cli_export_individual_flags(tmp_path):
    pandas = pytest.importorskip("pandas")
    from reqgraph.__main__ import main
    csv_in = tmp_path / "in.csv"
    csv_in.write_text("id,text\nR1,The system shall log.\n", encoding="utf-8")
    rc = main(["export", str(csv_in),
               "--csv", str(tmp_path / "q.csv"),
               "--graphml", str(tmp_path / "g.graphml")])
    assert rc == 0
    assert (tmp_path / "q.csv").exists()
    assert (tmp_path / "g.graphml").exists()
    assert not (tmp_path / "q.json").exists()


def test_cli_export_json_input(tmp_path):
    import json as _json
    pandas = pytest.importorskip("pandas")
    from reqgraph.__main__ import main
    data = [{"id": rid, "text": text} for rid, text in SET_REQS]
    json_in = tmp_path / "in.json"
    json_in.write_text(_json.dumps(data), encoding="utf-8")
    rc = main(["export", str(json_in), "--csv", str(tmp_path / "out.csv")])
    assert rc == 0
    assert (tmp_path / "out.csv").exists()


# --- GUI export_request ------------------------------------------------------

def test_gui_export_request_payload(tmp_path):
    import csv as _csv, io as _io
    from reqgraph.gui import GuiState, export_request
    buf = _io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["id", "text", "rationale"])
    for rid, text in SET_REQS:
        w.writerow([rid, text, f"rationale for {rid}"])
    csv_content = buf.getvalue()
    d = export_request(GuiState(), {"format": "csv", "content": csv_content})
    assert d["n_requirements"] == 4
    assert d["n_connections"] >= 0
    assert {c["role"] for c in d["connections"]} <= {"SUBJECT", "OBJECT"}
    assert "<graphml" in d["graphml"]
    assert "ELEMENT" in d["graphml"]   # element-level, not just requirement-level
    assert "HAS_ELEMENT" in d["graphml"]
    assert "csv_data" in d
    assert "rationale" in d["requirements"][0]   # metadata preserved


def test_gui_export_request_rejects_empty():
    from reqgraph import ReqGraphError
    from reqgraph.gui import GuiState, export_request
    with pytest.raises(ReqGraphError):
        export_request(GuiState(), {"format": "csv", "content": "   "})
