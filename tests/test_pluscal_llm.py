"""
Tests for requirement → PlusCal formalisation, the Ollama / Vertex AI providers
and the guarded AI-assist features.

TLC-backed tests run when tla2tools.jar is found (``$RAVEN_TLA2TOOLS`` or the
usual install locations) and a Java runtime is on PATH; otherwise they skip.
Provider tests talk to local fake servers speaking the real wire protocols, so
they exercise request/response shapes without network access or credentials.
"""

import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reqgraph import RequirementParser, RUPP_TEMPLATE, Role
from reqgraph.errors import ReqGraphError
from reqgraph.pluscal import (find_tla_tools, requirement_to_pluscal,
                              requirements_to_pluscal, validate_spec)

CASES = {
    "event_deadline": ("REQ-002", "When the cabin altitude exceeds 14,000 feet, the "
                       "oxygen system shall deploy the passenger oxygen masks within 4 seconds."),
    "ubiquitous": ("REQ-001", "The flight management system shall calculate the "
                   "optimal cruise altitude."),
    "compound": ("REQ-004", "As soon as an engine fire is detected, the engine control "
                 "unit shall shut off the affected engine and activate the fire "
                 "suppression system within 500 milliseconds."),
    "prohibition_state": ("REQ-005", "The system shall not transmit the message "
                          "during radio silence."),
    "optional": ("REQ-006", "Where dual redundancy is installed, the system should "
                 "report the channel status."),
    "unwanted": ("REQ-007", "If the airspeed drops below the stall speed, then the "
                 "stall warning system shall activate the stick shaker."),
    "permitted": ("REQ-009", "The operator may acknowledge the alarm."),
    "prohibition_always": ("REQ-010", "The system shall not transmit unencrypted data."),
    "decimal_deadline": ("REQ-011", "The pump shall start within 4.5 seconds."),
}


def _spec(key):
    rid, text = CASES[key]
    return requirement_to_pluscal(text, req_id=rid)


def _checked(spec):
    return {p["name"] for p in spec.properties if p["checked"]}


# --- generation (no external tools) ----------------------------------------

@pytest.mark.parametrize("key,pattern,obligation,props", [
    ("event_deadline", "event", "mandatory", {"TypeOK", "Response", "DeadlineMet"}),
    ("ubiquitous", "ubiquitous", "mandatory", {"TypeOK", "Response"}),
    ("compound", "event", "mandatory", {"TypeOK", "Response", "DeadlineMet"}),
    ("prohibition_state", "state", "prohibited", {"TypeOK", "Prohibition"}),
    ("optional", "optional", "advisory", {"TypeOK", "Response"}),
    ("unwanted", "unwanted", "mandatory", {"TypeOK", "Response"}),
    ("permitted", "ubiquitous", "permitted", {"TypeOK"}),
    ("prohibition_always", "ubiquitous", "prohibited", {"TypeOK", "Never"}),
])
def test_pattern_obligation_and_properties(key, pattern, obligation, props):
    s = _spec(key)
    assert (s.pattern, s.obligation) == (pattern, obligation)
    assert _checked(s) == props
    # every checked property is named in the TLC config and defined in the module
    for name in props:
        assert re.search(rf"^\s*{name}\s*==", s.tla, re.M), name
        assert name in s.cfg
    assert "CHECK_DEADLOCK FALSE" in s.cfg
    assert s.tla.lstrip().startswith("-") and s.tla.rstrip().endswith("=" * 77)


def test_identifiers_are_traceable_and_valid():
    s = _spec("event_deadline")
    assert s.module == "REQ_002"
    by_role = {m["role"]: m for m in s.mapping}
    # "14,000" loses its thousands separator; articles are dropped
    assert by_role["CONDITION"]["spec"] == "cabin_altitude_exceeds_14000_feet"
    assert by_role["PROCESS+OBJECT"]["spec"] == "deploy_passenger_oxygen_masks"
    assert by_role["SUBJECT"]["spec"] == "process OxygenSystem"
    assert by_role["CONSTRAINT"]["spec"] == "Deadline = 4"
    assert s.constants == {"Deadline": 4, "MaxTime": 5}
    # every variable is declared once
    decls = re.findall(r"^\s{4}(\w+)\s*(?:=|\\in)", s.tla.split("define")[0], re.M)
    assert len(decls) == len(set(decls))


def test_compound_and_actions_both_required():
    s = _spec("compound")
    acts = [m["spec"] for m in s.mapping if m["role"] == "PROCESS+OBJECT"]
    assert len(acts) == 2
    assert f"Done == ({acts[0]} /\\ {acts[1]})" in s.tla


def test_deadline_units_and_non_temporal_constraints():
    assert _spec("compound").constants["Deadline"] == 500          # milliseconds
    dec = _spec("decimal_deadline")
    assert dec.constants["Deadline"] == 45                        # 45 x 0.1 seconds
    assert "0.1 seconds" in dec.tla
    # "2 m" is a distance, not two minutes -- it must not become a deadline
    s = requirement_to_pluscal("The drone shall maintain altitude within 2 m.")
    assert "DeadlineMet" not in _checked(s)
    assert any("not formalised" in n and "2 m" in n for n in s.notes)


def test_incomplete_requirement_is_flagged_not_hidden():
    s = requirement_to_pluscal("The drone shall log TBD events.", req_id="SYS-3")
    assert any("incomplete" in n and "TBD" in n for n in s.notes)


def test_no_action_cannot_be_formalised():
    with pytest.raises(ReqGraphError, match="no action"):
        requirement_to_pluscal("The system shall.")


def test_no_modality_is_not_silently_made_mandatory():
    """Regression: a fragment with no shall/should/may used to be formalised
    as a *mandatory* obligation -- inventing a commitment the text never made."""
    with pytest.raises(ReqGraphError, match="no modality"):
        requirement_to_pluscal("The system.")


def test_reserved_words_and_module_names():
    s = requirement_to_pluscal("The pump shall process the request.", req_id="42")
    assert s.module == "REQ_42"
    s2 = requirement_to_pluscal("The clock shall display the time.", req_id="R-9")
    assert s2.module == "R_9"
    assert "clock" not in [m["spec"] for m in s2.mapping if m["role"] == "PROCESS+OBJECT"]


def test_batch_keeps_going_past_unformalisable_items():
    out = requirements_to_pluscal([("A", CASES["ubiquitous"][1]), ("B", "The system.")])
    assert out[0].module == "A"
    assert out[1][0] == "B" and "no modality" in out[1][1]


def test_leading_where_is_a_condition_but_midsentence_where_is_not():
    p = RequirementParser(RUPP_TEMPLATE)
    g = p.parse("Where dual redundancy is installed, the system should report the status.")
    assert g.by_role(Role.CONDITION)[0].text == "Where dual redundancy is installed"
    text = "The display shall show the bay where the drone is parked."
    g2 = p.parse(text)
    assert not g2.by_role(Role.CONDITION)
    assert g2.generate() == text


def test_validate_without_tools_reports_instead_of_raising(monkeypatch):
    monkeypatch.setenv("RAVEN_TLA2TOOLS", "/nonexistent/tla2tools.jar")
    monkeypatch.setattr("reqgraph.pluscal._JAR_CANDIDATES", ())
    v = validate_spec(_spec("ubiquitous"))
    assert v["available"] is False and v["ok"] is False and v["error"]


# --- TLC-backed: reference models pass, broken designs are caught ------------

_TLA = find_tla_tools()
needs_tlc = pytest.mark.skipif(
    not (_TLA and __import__("shutil").which("java")),
    reason="tla2tools.jar / java not available (set RAVEN_TLA2TOOLS)")


@needs_tlc
@pytest.mark.parametrize("key", sorted(CASES))
def test_reference_model_passes_tlc(key):
    v = validate_spec(_spec(key))
    assert v["translated"], v
    assert v["ok"], (v["violations"], v["error"])


def _mutate(tla, how):
    if how == "slow":        # time may run past the deadline
        return re.sub(r"\n\s*\\\* time may not pass.*\n\s*await ~\(Pending.*;", "", tla)
    if how == "lazy":        # the subject is not obliged to act
        return re.sub(r"^fair process ", "process ", tla, flags=re.M)
    if how == "violator":    # acts regardless of the prohibition
        return tla.replace("await ~Trigger; ", "")
    if how == "always":
        var = re.search(r"^\s{4}(\w+) = FALSE", tla, re.M).group(1)
        return tla.replace("skip;   \\* a compliant system never takes the action",
                           f"{var} := TRUE;")
    raise ValueError(how)


@needs_tlc
@pytest.mark.parametrize("key,how,expect", [
    ("event_deadline", "slow", "DeadlineMet"),
    ("event_deadline", "lazy", "Response"),
    ("compound", "slow", "DeadlineMet"),
    ("unwanted", "lazy", "Response"),
    ("optional", "lazy", "Response"),
    ("prohibition_state", "violator", "Prohibition"),
    ("prohibition_always", "always", "Never"),
])
def test_each_property_catches_the_defect_it_exists_for(key, how, expect):
    """A spec that always passes proves nothing: break the design, expect TLC to object."""
    s = _spec(key)
    v = validate_spec(_mutate(s.tla, how), cfg=s.cfg)
    assert v["checked"], v
    assert expect in {x["name"] for x in v["violations"]}, v["violations"]


@needs_tlc
def test_bounded_response_state_space_stays_small():
    """Time is measured from the trigger, so a 500 ms deadline stays tractable."""
    v = validate_spec(_spec("compound"))
    assert v["ok"] and v["distinct_states"] < 10_000


# --- LLM providers over real HTTP (local fakes) -----------------------------

class _FakeLLM(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def _reply(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        _FakeLLM.seen.append(("GET", self.path, None, dict(self.headers)))
        if self.path == "/api/tags":
            return self._reply({"models": [{"name": "llama3.1:latest"}]})
        self._reply({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeLLM.seen.append(("POST", self.path, body, dict(self.headers)))
        if self.path == "/api/chat":
            return self._reply({"message": {"role": "assistant",
                                            "content": '{"answer": 42}'}, "done": True})
        if self.path.endswith(":generateContent"):
            if self.headers.get("Authorization") != "Bearer good-token":
                return self._reply({"error": {"code": 401, "message": "bad token"}}, 401)
            return self._reply({"candidates": [{"content": {"parts": [
                {"text": "hello "}, {"text": "gemini"}]}}]})
        self._reply({"error": "not found"}, 404)


@pytest.fixture
def fake_llm():
    _FakeLLM.seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_ollama_chat_protocol_and_status(fake_llm):
    from reqgraph.llm import OllamaProvider, extract_json
    o = OllamaProvider(model="llama3.1", host=fake_llm)
    st = o.status()
    assert st["available"] and st["reachable"]
    assert extract_json(o.complete("q", system="s", json_mode=True)) == {"answer": 42}
    _, path, body, _ = _FakeLLM.seen[-1]
    assert path == "/api/chat" and body["stream"] is False and body["format"] == "json"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    missing = OllamaProvider(model="phi3", host=fake_llm).status()
    assert not missing["available"] and "ollama pull phi3" in missing["detail"]


def test_ollama_bypasses_a_corporate_proxy_for_localhost(fake_llm, monkeypatch):
    from reqgraph.llm import OllamaProvider
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")    # a dead proxy
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert OllamaProvider(host=fake_llm).status()["reachable"] is True


def test_ollama_unreachable_is_a_clear_status_not_a_crash():
    from reqgraph.llm import OllamaProvider
    st = OllamaProvider(host="127.0.0.1:1").status()
    assert st["available"] is False and "ollama serve" in st["detail"]


@pytest.mark.parametrize("raw,expected", [
    ("0.0.0.0:11434", "http://127.0.0.1:11434"),
    ("localhost", "http://localhost:11434"),
    ("http://gpu-box:8080/", "http://gpu-box:8080"),
])
def test_ollama_host_normalisation(raw, expected):
    from reqgraph.llm import OllamaProvider
    assert OllamaProvider(host=raw).host == expected


def test_vertex_generatecontent_protocol(fake_llm):
    from reqgraph.llm import LLMError, VertexAIProvider
    v = VertexAIProvider(model="gemini-x", project="proj", location="europe-west9",
                         access_token="good-token", endpoint=f"http://{fake_llm}")
    assert v.complete("q", system="sys", json_mode=True) == "hello gemini"
    _, path, body, headers = _FakeLLM.seen[-1]
    assert path == ("/v1/projects/proj/locations/europe-west9/publishers/google/"
                    "models/gemini-x:generateContent")
    assert body["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    bad = VertexAIProvider(model="gemini-x", project="proj", location="europe-west9",
                           access_token="nope", endpoint=f"http://{fake_llm}")
    with pytest.raises(LLMError, match="401.*bad token"):
        bad.complete("q")


def test_vertex_regional_and_global_endpoints():
    from reqgraph.llm import VertexAIProvider
    assert VertexAIProvider(project="p", location="europe-west9", access_token="t").url() \
        .startswith("https://europe-west9-aiplatform.googleapis.com/v1/projects/p/")
    assert VertexAIProvider(project="p", location="global", access_token="t").url() \
        .startswith("https://aiplatform.googleapis.com/v1/projects/p/locations/global/")


def test_vertex_credential_sources(monkeypatch):
    import builtins
    from reqgraph.llm import LLMError, VertexAIProvider
    monkeypatch.setenv("GOOGLE_OAUTH_ACCESS_TOKEN", "from-env")
    assert VertexAIProvider(project="p")._access_token() == "from-env"
    # nothing available -> an actionable error, not a stack trace
    monkeypatch.delenv("GOOGLE_OAUTH_ACCESS_TOKEN")
    monkeypatch.setattr("reqgraph.llm.shutil.which", lambda _: None)
    real_import = builtins.__import__

    def no_google(name, *a, **kw):
        if name.startswith("google"):
            raise ImportError(name)
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", no_google)
    with pytest.raises(LLMError, match="gcloud auth application-default login"):
        VertexAIProvider(project="p")._access_token()


def test_provider_factory(monkeypatch):
    from reqgraph.llm import LLMError, OllamaProvider, VertexAIProvider, get_provider
    monkeypatch.delenv("RAVEN_LLM_PROVIDER", raising=False)
    assert get_provider() is None and get_provider("off") is None
    assert isinstance(get_provider("ollama", host=""), OllamaProvider)   # blanks ignored
    assert isinstance(get_provider("gcp", project="p", access_token="t"), VertexAIProvider)
    with pytest.raises(LLMError, match="unknown LLM provider"):
        get_provider("gpt")


@pytest.mark.parametrize("reply", [
    '{"a": 1}', 'Sure!\n```json\n{"a": 1}\n```', 'Here you go: {"a": 1} -- enjoy',
])
def test_extract_json_tolerates_model_formatting(reply):
    from reqgraph.llm import extract_json
    assert extract_json(reply) == {"a": 1}


# --- AI assist: candidates are re-checked, weakening is rejected -------------

class _Scripted:
    name, model = "scripted", "test"

    def __init__(self, reply):
        self.reply = reply

    def complete(self, prompt, **kw):
        self.prompt = prompt
        return self.reply

    def describe(self):
        return "scripted:test"


@pytest.mark.parametrize("text,rewrite,verdict", [
    ("It shall be closed automatically.", "The fuel valve shall close automatically.",
     "improved"),
    ("The system shall respond quickly.",
     "The system shall respond to an operator command within <value> milliseconds.",
     "needs_input"),
    ("The pump shall start within 4 seconds.", "The pump shall start quickly.",
     "regressed"),
    ("The pump shall start within 4 seconds.", "The pump shall start within 4 seconds.",
     "unchanged"),
])
def test_rewrite_is_rescored_by_raven(text, rewrite, verdict):
    from reqgraph.assist import suggest_rewrite
    prov = _Scripted(json.dumps({"rewrite": [rewrite], "rationale": "r"}))
    r = suggest_rewrite(text, prov)
    assert r["status"] == "candidate"
    assert r["verdict"] == verdict, (r["resolved"], r["introduced"])
    assert "Findings" in prov.prompt or "no defects" in prov.prompt


def _refine_reply(tla, cfg):
    return f"```tla\n{tla}```\n```cfg\n{cfg}```\nrationale"


def test_refine_accepts_an_honest_refinement():
    from reqgraph.assist import refine_pluscal
    s = _spec("event_deadline")
    honest = s.tla.replace("  Raise:", "  Raise:   \\* the pressure sensor reports it")
    r = refine_pluscal(s, _Scripted(_refine_reply(honest, s.cfg)), validate=False)
    assert r["accepted"] and not r["reasons"]


@pytest.mark.parametrize("attack", ["drop_property", "redefine_helper", "no_code"])
def test_refine_rejects_weakening(attack):
    """A model can make TLC pass by checking less; that must never be accepted."""
    from reqgraph.assist import refine_pluscal
    s = _spec("event_deadline")
    if attack == "drop_property":
        reply = _refine_reply(s.tla, s.cfg.replace("PROPERTIES Response\n", ""))
        why = "dropped required property: Response"
    elif attack == "redefine_helper":     # Response untouched, but Done made trivial
        reply = _refine_reply(s.tla.replace("Done == deploy_passenger_oxygen_masks",
                                            "Done == TRUE"), s.cfg)
        why = "changed the definition of Done"
    else:
        reply = "I cannot help with that."
        why = "no TLA+ module"
    r = refine_pluscal(s, _Scripted(reply), validate=False)
    assert r["accepted"] is False
    assert any(why in reason for reason in r["reasons"]), r["reasons"]


# --- GUI: endpoints + same-origin hardening -----------------------------------

@pytest.fixture
def gui():
    from reqgraph.gui import make_server
    srv = make_server(port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def _post(port, path, body, headers=None):
    hdrs = {"Content-Type": "application/json"} if headers is None else headers
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(), headers=hdrs,
                                 method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


@pytest.mark.parametrize("headers", [
    {"Content-Type": "text/plain"},                                   # no-preflight CSRF
    {"Content-Type": "application/json", "Origin": "https://evil.example"},
])
def test_gui_rejects_cross_site_posts(gui, headers):
    code, d = _post(gui, "/api/parse", {"text": "The system shall run."}, headers)
    assert code == 403 and d["error"]


def test_gui_pluscal_and_ai_endpoints(gui):
    code, d = _post(gui, "/api/pluscal", {"text": CASES["event_deadline"][1],
                                          "id": "REQ-002"})
    assert code == 200 and d["module"] == "REQ_002" and "--algorithm" in d["tla"]
    assert {p["name"] for p in d["properties"] if p["checked"]} == \
        {"TypeOK", "Response", "DeadlineMet"}
    code, d = _post(gui, "/api/llm/rewrite", {"text": "x", "llm": {"provider": "none"}})
    assert code == 400 and "AI assistant is off" in d["error"]
    code, d = _post(gui, "/api/llm/status",
                    {"llm": {"provider": "vertex", "project": "p",
                             "endpoint": "http://evil.example"}})
    assert code == 400 and "https" in d["error"]
    info = json.loads(urllib.request.build_opener(urllib.request.ProxyHandler({}))
                      .open(f"http://127.0.0.1:{gui}/api/info", timeout=10).read())
    assert "available" in info["tla_tools"]
    assert info["llm_default"]["ollama"]["model"]
    assert "token" not in json.dumps(info["llm_default"]).lower()     # never secrets


# --- CLI behaviours documented in the top-level README ----------------------

def test_cli_export_creates_missing_output_folders(tmp_path):
    """README example `export … --out-prefix build/out` must work on a fresh
    checkout (it used to crash with 'Cannot save file into a non-existent
    directory')."""
    pytest.importorskip("pandas")
    from reqgraph.__main__ import main
    src = tmp_path / "reqs.csv"
    src.write_text("id,text\nR1,The pump shall start within 4 seconds.\n", encoding="utf-8")
    prefix = tmp_path / "build" / "nested" / "out"
    assert main(["export", str(src), "--out-prefix", str(prefix)]) == 0
    for ext in (".csv", ".json", ".graphml", ".req.ttl"):
        assert (tmp_path / "build" / "nested" / f"out{ext}").exists(), ext


def test_cli_ai_commands_fail_once_when_provider_not_ready(tmp_path, capsys):
    """An unreachable provider stops with one actionable error and exit code != 0,
    instead of repeating the same failure for every requirement."""
    from reqgraph.__main__ import main
    src = tmp_path / "reqs.txt"
    src.write_text("The system shall respond quickly.\nThe pump shall start.\n",
                   encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", str(src), "--provider", "ollama", "--host", "127.0.0.1:1"])
    assert "ollama is not ready" in str(exc.value) and "ollama serve" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        main(["pluscal", str(src), "--out", str(tmp_path / "specs"), "--refine",
              "--provider", "ollama", "--host", "127.0.0.1:1"])
    assert "ollama is not ready" in str(exc.value)
