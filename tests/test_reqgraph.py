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


# --- batch I/O -------------------------------------------------------------

def test_reqif_roundtrip():
    from reqgraph.io_formats import write_reqif, read_reqif
    items = [(f"REQ-{i+1:03d}", t) for i, t in enumerate(REQS)]
    with tempfile.TemporaryDirectory() as d:
        path = write_reqif(items, os.path.join(d, "r.reqif"))
        back = read_reqif(path)
    assert [t for _, t in back] == [t for _, t in items]


def test_csv_roundtrip():
    from reqgraph.io_formats import write_csv, read_requirements_csv
    items = [(f"REQ-{i+1:03d}", t) for i, t in enumerate(REQS)]
    with tempfile.TemporaryDirectory() as d:
        path = write_csv(items, os.path.join(d, "r.csv"))
        back = read_requirements_csv(path)
    assert [t for _, t in back] == [t for _, t in items]


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


def test_gui_parse_request_rejects_empty():
    from reqgraph import ReqGraphError
    from reqgraph.gui import GuiState, parse_request
    with pytest.raises(ReqGraphError):
        parse_request(GuiState(), {"text": "   "})


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
        req = urllib.request.Request(
            f"{base}/api/parse",
            data=_json.dumps({"text": REQS[0]}).encode(),
            headers={"Content-Type": "application/json"})
        out = _json.load(urllib.request.urlopen(req, timeout=10))
        assert out["kpis"]["roundtrip_ok"] is True
    finally:
        server.shutdown()
        server.server_close()
