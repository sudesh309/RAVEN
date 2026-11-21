"""Tests for the agentic review: architecture extraction, every agent, the
orchestrator, LangGraph wiring, tool specs, MCP server, CLI and GUI endpoint."""

import io
import json
import os

import pytest

from reqgraph.architecture import (ArchitectureModel, from_sysml_v1, load_architecture,
                                   parse_architecture_sysml)
from reqgraph.agents import (AllocationAgent, ArchitectureJustificationAgent, CascadeAgent,
                             CompletenessAgent, ConsistencyAgent, DerivedRequirementsAgent,
                             RationaleAgent, ReviewContext, ReviewOrchestrator,
                             WritingQualityAgent, default_agents, review)
from reqgraph.errors import DataFormatError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRONE_REQIF = os.path.join(ROOT, "samples", "drone", "drone_requirements.reqif")
DRONE_SYSML = os.path.join(ROOT, "samples", "drone", "drone_model.sysml")

MODEL = """
package Rover {
  part def Rover {
    doc /* The rover system. */
    part drive : DriveUnit;
    part nav : NavComputer;
    part radio : Radio;
  }
  part def DriveUnit {
    part motor : Motor;
    part gearbox : Gearbox;
    port power : PowerPort;
  }
  part def Motor;
  part def Gearbox;
  part def NavComputer { satisfy requirement NavReq; }
  part def Radio;
  part def Heater;
  port def PowerPort;
  requirement <'ROV-1'> SpeedReq {
    doc /* The rover shall travel at no more than 2 m/s. */
    subject rover : Rover;
  }
  requirement NavReq { doc /* The nav computer shall plan a route within 5 s. */ }
  requirement ThermalReq { doc /* The rover shall survive -40 C nights. */ }
  satisfy SpeedReq by drive;
  allocate radio to Radio;
  verification def SpeedTest { verify requirement SpeedReq; }
  #derivation connection {
    end #original ::> SpeedReq;
    end #derive ::> NavReq;
  }
}
"""


def _ids(findings, rule):
    return sorted(f.subject_id for f in findings if f.rule == rule)


def _run(agent, reqs, arch=None, config=None):
    a = arch.to_dict() if isinstance(arch, ArchitectureModel) else arch
    return agent.run(ReviewContext(reqs, a, config))


# ---------------------------------------------------------------- architecture
class TestArchitecture:
    def test_parse_structure_requirements_links(self):
        m = parse_architecture_sysml(MODEL)
        assert m.children("Rover") == ["DriveUnit", "NavComputer", "Radio"]
        assert set(m.children("DriveUnit")) == {"Motor", "Gearbox", "PowerPort"}
        assert m.depth()["Motor"] == 2
        assert m.descendants("Rover") >= {"Motor", "Gearbox"}
        r = m.requirement("ROV-1")
        assert r.name == "SpeedReq" and "2 m/s" in r.text and r.subjects == ["Rover"]
        kinds = {(l.kind, l.source, l.target) for l in m.links}
        assert ("satisfy", "drive", "SpeedReq") in kinds
        assert ("satisfy", "NavComputer", "NavReq") in kinds        # satisfy inside a part
        assert ("allocate", "radio", "Radio") in kinds
        assert ("verify", "SpeedTest", "SpeedReq") in kinds
        assert ("derive", "NavReq", "SpeedReq") in kinds
        assert m.resolve("drive") == "DriveUnit" and m.resolve("Rover::nav") == "NavComputer"
        assert m.definitions()[0].doc == "The rover system."

    def test_round_trip_and_summary(self):
        m = parse_architecture_sysml(MODEL)
        m2 = ArchitectureModel.from_dict(json.loads(json.dumps(m.to_dict())))
        assert m2.summary() == m.summary()

    def test_drone_sample(self):
        m = load_architecture(DRONE_SYSML)
        assert len(m.requirements) == 7
        assert all(r.subjects for r in m.requirements)
        assert m.depth()["AttitudeController"] == 2
        assert "FlightControlUnit" in m.children("DroneSystem")

    def test_empty_and_garbage(self):
        with pytest.raises(DataFormatError):
            parse_architecture_sysml("   ")
        with pytest.raises(DataFormatError):
            parse_architecture_sysml("just some prose, no model here")

    def test_v1_adapter(self):
        from reqgraph.sysml_v1_parser import SysMLV1Model, V1Element, V1Relation
        from reqgraph.core import Role
        els = [V1Element("1", "Aircraft", "uml:Class", "Block", Role.SUBJECT),
               V1Element("2", "Engine", "uml:Class", "Block", Role.SUBJECT),
               V1Element("3", "Thrust", "uml:Class", "Requirement", Role.SUBJECT,
                         req_text="The engine shall deliver 100 kN.", req_id="REQ-7")]
        rels = [V1Relation("r1", "1", "2", "composition", name="engine"),
                V1Relation("r2", "2", "3", "satisfy")]
        m = from_sysml_v1(SysMLV1Model(els, rels))
        assert m.children("Aircraft") == ["Engine"]
        assert m.requirement("REQ-7").text.startswith("The engine")
        assert ("satisfy", "Engine", "Thrust") in {(l.kind, l.source, l.target) for l in m.links}


# ---------------------------------------------------------------- quality team
class TestQualityAgents:
    def test_writing_quality(self):
        r = _run(WritingQualityAgent(), [("A", "The system should be fast and user-friendly and reliable.", {}),
                                         ("B", "The pump shall deliver 5 l/min.", {})])
        assert "A" in _ids(r.findings, "weak_words") and "A" in _ids(r.findings, "non_atomic")
        assert not [f for f in r.findings if f.subject_id == "B"]
        assert r.metrics["pct_clean"] == 50.0

    def test_completeness_set_coverage(self):
        reqs = [(f"R{i}", f"The pump shall deliver {i} l/min within 2 s.", {}) for i in range(1, 7)]
        r = _run(CompletenessAgent(), reqs)
        rules = {f.rule for f in r.findings}
        assert "no_off_nominal" in rules and "no_safety_requirements" in rules

    def test_rationale(self):
        reqs = [("A", "The valve shall close within 2 s.", {}),
                ("B", "The valve shall close within 2 s.", {"rationale": "TBD", "verification": "Test"}),
                ("C", "The valve shall close within 2 s.",
                 {"rationale": "The valve shall close within 2 s.", "source": "SRD-1"}),
                ("D", "The valve shall close within 2 s.",
                 {"rationale": "Limits fuel spill to 0.5 l per the fire hazard analysis FHA-12.",
                  "verification_method": "Inspection", "parent": "X"}),
                ("E", "If a fire is detected, the valve shall close.", {})]
        r = _run(RationaleAgent(), reqs)
        assert _ids(r.findings, "missing_rationale") == ["A", "E"]
        assert [f.severity for f in r.findings if f.subject_id == "E" and
                f.rule == "missing_rationale"] == ["blocker"]           # safety requirement
        assert _ids(r.findings, "trivial_rationale") == ["B"]
        assert _ids(r.findings, "restating_rationale") == ["C"]
        assert "D" in _ids(r.findings, "weak_verification_method")
        assert "A" in _ids(r.findings, "missing_verification_method")
        assert "D" not in _ids(r.findings, "missing_source")

    def test_consistency(self):
        reqs = [("A", "The heater shall activate when the temperature is below 5 C.", {}),
                ("a", "The heater shall activate when temperature is below 5 C.", {}),
                ("C", "The pump shall start the engine within 2 s.", {}),
                ("D", "The pump shall not start the engine within 2 s.", {}),
                ("E", "The display shall refresh at 60 Hz in normal mode at night.", {}),
                ("F", "The display shall refresh at 30 Hz in normal mode at night.", {})]
        r = _run(ConsistencyAgent(), reqs)
        assert "A" in _ids(r.findings, "duplicate_id") + _ids(r.findings, "duplicate") or \
            "a" in _ids(r.findings, "duplicate_id")
        assert _ids(r.findings, "polarity_conflict") == ["C"]
        assert _ids(r.findings, "value_conflict") == ["E"]


# ---------------------------------------------------------------- cascade
class TestCascade:
    REQS = [("S1", "The vehicle shall reach the destination within 30 minutes.", {"level": "1"}),
            ("S2", "The vehicle shall carry 4 passengers.", {"level": "1"}),
            ("A1", "The drive unit shall propel the vehicle to the destination within 40 minutes.",
             {"level": "2", "parent": "S1"}),
            ("A2", "The battery shall store 50 kWh.", {"level": "2"}),
            ("A3", "The seat shall fold.", {"level": "2", "parent": "S9"}),
            ("B1", "The motor shall deliver 100 kW.", {"level": "1", "parent": "A1"}),
            ("C1", "The x shall do a.", {"level": "3", "parent": "C2"}),
            ("C2", "The y shall do b.", {"level": "3", "parent": "C1"})]

    def test_structure_and_budgets(self):
        r = _run(CascadeAgent(), self.REQS)
        assert r.status == "ok"
        assert _ids(r.findings, "dangling_parent") == ["A3"]
        assert _ids(r.findings, "cycle") == ["C1", "C2"]
        assert "A2" in _ids(r.findings, "orphan")
        assert "B1" in _ids(r.findings, "level_inversion")
        assert "S2" in _ids(r.findings, "not_decomposed")
        assert _ids(r.findings, "budget_violation") == ["A1"]           # 40 min > 30 min
        assert r.data["tree"]["children"]["S1"] == ["A1"]

    def test_skipped_without_hierarchy(self):
        r = _run(CascadeAgent(), [("A", "The pump shall run.", {})])
        assert r.status == "skipped" and "hierarchy" in r.reason

    def test_reqif_links_feed_the_cascade(self):
        from reqgraph.io_formats import read_reqif
        items = read_reqif(DRONE_REQIF)
        parents = {rid: m.get("parent") for rid, _, m in items}
        assert parents["DRONE-FC-001"] == "DRONE-SYS-001"
        r = _run(CascadeAgent(), items)
        assert r.metrics["links"] == 36 and r.metrics["levels"] == [1, 2, 3, 4]
        assert not _ids(r.findings, "dangling_parent")


# ---------------------------------------------------------------- architecture team
class TestArchitectureAgents:
    REQS = [("ROV-1", "The rover shall travel at no more than 2 m/s.", {}),
            ("R2", "The navigation computer shall plan a route within 5 s.", {"parent": "ROV-1"}),
            ("R3", "The camera shall record 4K video.", {}),
            ("R4", "The radio shall transmit at 1 W.", {"allocated_to": "Antenna"}),
            ("R5", "The motor shall deliver 50 W.", {"parent": "R2"})]

    def setup_method(self):
        self.m = parse_architecture_sysml(MODEL)

    def test_allocation(self):
        r = _run(AllocationAgent(), self.REQS, self.m)
        alloc = r.data["allocations"]
        assert sorted(alloc["ROV-1"]["explicit"]) == ["DriveUnit", "Rover"]      # satisfy-by + subject
        assert alloc["ROV-1"]["model_requirement"] == "SpeedReq"
        assert alloc["R2"]["explicit"] == ["NavComputer"]                # matched NavReq by text
        assert _ids(r.findings, "unallocated") == ["R3"]
        assert _ids(r.findings, "unknown_allocation_target") == ["R4"]
        assert "R5" in _ids(r.findings, "implicit_allocation")

    def test_justification(self):
        r = _run(ArchitectureJustificationAgent(), self.REQS, self.m)
        assert _ids(r.findings, "unjustified_element") == ["Heater"]
        assert all(f.subject_kind == "element" for f in r.findings)

    def test_derived_requirements(self):
        r = _run(DerivedRequirementsAgent(), self.REQS, self.m)
        assert _ids(r.findings, "model_requirement_missing") == ["ThermalReq"]
        assert "Gearbox" in _ids(r.findings, "derived_requirement_expected")
        assert "Motor" not in _ids(r.findings, "derived_requirement_expected")   # R5 covers it
        # R5 (Motor, under DriveUnit) traces to R2 (NavComputer): wrong branch
        assert _ids(r.findings, "allocation_outside_parent_branch") == ["R5"]

    def test_skipped_without_model(self):
        r = _run(AllocationAgent(), self.REQS)
        assert r.status == "skipped"


# ---------------------------------------------------------------- orchestration
class TestOrchestrator:
    def test_plan_and_report(self):
        rep = review([("A", "The pump shall deliver 5 l/min.", {})])
        s = rep["summary"]
        assert s["plan"] == ["writing_quality", "completeness", "rationale", "consistency"]
        assert set(s["agents_skipped"]) == {"cascade", "allocation", "architecture_justification",
                                            "derived_requirements"}
        assert rep["requirements"][0]["status"] == "needs work"
        md, csv_ = rep.to_markdown(), rep.to_csv()
        assert "# RAVEN agentic requirement review" in md and csv_.startswith("severity,")

    def test_drone_full_review(self):
        from reqgraph.io_formats import read_reqif
        rep = review(read_reqif(DRONE_REQIF), load_architecture(DRONE_SYSML))
        s = rep["summary"]
        assert len(s["agents_run"]) == 8 and not s["agents_failed"]
        assert rep["metrics"]["allocation"]["pct_allocated"] > 70
        assert rep["elements"]

    def test_team_selection(self):
        assert [a.name for a in default_agents(names="architecture")] == [
            "allocation", "architecture_justification", "derived_requirements"]
        with pytest.raises(ValueError):
            default_agents(names=["nope"])

    def test_agent_failure_is_contained(self):
        class Boom(WritingQualityAgent):
            name = "boom"

            def analyze(self, ctx, result):
                raise RuntimeError("kaput")
        rep = ReviewOrchestrator([Boom(), RationaleAgent()]).run([("A", "x shall y.", {})])
        assert rep["summary"]["agents_failed"] == {"boom": "RuntimeError: kaput"}
        assert "rationale" in rep["summary"]["agents_run"]


class ScriptedLLM:
    name, model = "scripted", "test"

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def complete(self, prompt, *, system=None, temperature=0.2, json_mode=False):
        self.prompts.append(prompt)
        r = self.replies.pop(0) if self.replies else '{"findings": []}'
        if isinstance(r, Exception):
            raise r
        return r

    def describe(self):
        return "scripted:test"


class TestLLM:
    def test_candidates_guarded(self):
        llm = ScriptedLLM([json.dumps({"findings": [
            {"subject_id": "A", "rule": "Undefined Term", "severity": "blocker",
             "message": "'nominal' is undefined", "hint": "define it"},
            {"subject_id": "GHOST", "rule": "x", "severity": "major", "message": "hallucinated"}]})])
        r = WritingQualityAgent(llm=llm).run(
            ReviewContext([("A", "The pump shall run at nominal speed.", {})], config={"use_llm": True}))
        llm_f = [f for f in r.findings if f.source == "llm"]
        assert len(llm_f) == 1 and llm_f[0].status == "candidate"
        assert llm_f[0].severity == "minor"            # an LLM cannot raise a blocker
        assert llm_f[0].rule == "undefined_term"
        assert r.llm_calls == 1 and "nominal speed" in llm.prompts[0]

    def test_provider_failure_is_recorded(self):
        from reqgraph.llm import LLMError
        llm = ScriptedLLM([LLMError("ollama down")])
        rep = ReviewOrchestrator(default_agents(llm, "quality"), llm=llm,
                                 config={"use_llm": True}).run([("A", "The pump shall run.", {})])
        assert rep["agents"]["writing_quality"]["data"]["llm_errors"] == ["ollama down"]
        assert rep["agents"]["writing_quality"]["status"] == "ok"

    def test_call_cap(self):
        llm = ScriptedLLM([])
        reqs = [(f"R{i}", "The pump shall run.", {}) for i in range(40)]
        r = WritingQualityAgent(llm=llm).run(ReviewContext(reqs, config={"use_llm": True,
                                                                          "llm_max_calls": 2}))
        assert r.llm_calls == 2 and r.data["llm_truncated"]


# ---------------------------------------------------------------- integrations
class TestIntegrations:
    def test_langgraph(self):
        pytest.importorskip("langgraph")
        from reqgraph.agents.langgraph_adapter import build_review_graph, initial_state
        g = build_review_graph()
        out = g.invoke(initial_state(TestArchitectureAgents.REQS, parse_architecture_sysml(MODEL)))
        assert "allocation" in out["plan"] and "cascade" in out["plan"]
        assert set(out["results"]) == set(out["plan"])
        assert out["report"]["summary"]["architecture"] is True
        out = g.invoke(initial_state([("A", "The pump shall run.", {})]))
        assert "allocation" not in out["results"]

    def test_tools_and_mcp(self):
        from reqgraph.agents.mcp_server import handle, serve
        from reqgraph.agents.tools import TOOL_SPECS, call_tool
        assert {t["name"] for t in TOOL_SPECS} >= {"raven_review", "raven_architecture_review"}
        out = call_tool("raven_architecture_review",
                        {"requirements": [{"id": "R3", "text": "The camera shall record video."}],
                         "architecture_text": MODEL})
        assert out["summary"]["plan"] == ["allocation", "architecture_justification",
                                          "derived_requirements"]
        assert call_tool("raven_parse_architecture", {"architecture_text": MODEL})["summary"]["requirements"] == 3
        msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "raven_quality_review",
                            "arguments": {"requirements": [{"text": "The pump should be fast."}]}}},
                {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                 "params": {"name": "raven_review", "arguments": {}}},
                {"jsonrpc": "2.0", "id": 5, "method": "bogus"}]
        stdin = io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\nnot json\n")
        stdout = io.StringIO()
        serve(stdin, stdout)
        resp = [json.loads(l) for l in stdout.getvalue().splitlines()]
        assert [r.get("id") for r in resp] == [1, 2, 3, 4, 5, None]
        assert resp[0]["result"]["capabilities"]["tools"] is not None
        assert len(resp[1]["result"]["tools"]) == len(TOOL_SPECS)
        assert resp[2]["result"]["isError"] is False
        assert resp[2]["result"]["structuredContent"]["summary"]["requirements"] == 1
        assert resp[3]["result"]["isError"] is True
        assert resp[4]["error"]["code"] == -32601 and resp[5]["error"]["code"] == -32700
        assert handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                       "params": {"name": "nope"}})["error"]["code"] == -32602

    def test_cli(self, tmp_path, capsys):
        from reqgraph.__main__ import main
        out = tmp_path / "r.json"
        rc = main(["review", DRONE_REQIF, "-a", DRONE_SYSML, "--format", "json", "-o", str(out)])
        assert rc == 0
        d = json.loads(out.read_text())
        assert d["summary"]["architecture"] is True
        assert main(["review", DRONE_REQIF, "--agents", "quality", "--fail-on", "major"]) == 1

    def test_gui_endpoint(self):
        from reqgraph.gui import GuiState, review_request, review_sample_request
        st = GuiState()
        sample = review_sample_request(st, {})
        d = review_request(st, {"req_content": sample["req_content"], "req_format": "reqif",
                                "arch_content": sample["arch_content"]})
        assert d["summary"]["requirements"] == 40 and d["architecture_summary"]["requirements"] == 7
        assert d["markdown"].startswith("# RAVEN")
        from reqgraph.errors import ReqGraphError
        with pytest.raises(ReqGraphError):
            review_request(st, {"req_content": ""})
