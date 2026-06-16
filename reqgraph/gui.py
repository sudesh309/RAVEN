"""
reqgraph.gui
============

A zero-dependency local web GUI: stdlib ``http.server`` serving one
self-contained HTML page (``static/index.html``) plus a tiny JSON API.

    python -m reqgraph gui              # http://127.0.0.1:8765, opens browser

Endpoints
---------
GET  /            the single-page app
GET  /api/info    {templates, backends{name: available}, version}
POST /api/parse   {text, template?, backend?} ->
                  {kpis, tiles, tree, elements, mermaid, dot, graph, error?}

Design notes
------------
* Binds to 127.0.0.1 only -- this is a personal desktop tool, never exposed.
* Extractors are created lazily once and cached behind a lock (BERT/spaCy
  loads are expensive; ThreadingHTTPServer handles requests concurrently).
* ``parse_request`` is a pure function so the API logic is unit-testable
  without sockets.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from . import __version__
from .core import Rel, Role
from .errors import ReqGraphError
from .extractors import BertTaggerExtractor, RuleExtractor, SpacyExtractor
from .parser import RequirementParser
from .quality import enrich
from .templates import RUPP_TEMPLATE, TEMPLATES

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_BERT_DIR = os.path.join("models", "req_tagger")

# Human-readable "how it works" copy surfaced in the GUI so users understand
# what each configuration choice actually does. Served via /api/info.
BACKEND_INFO = {
    "rules": {
        "title": "Rule-based · deterministic",
        "how": "Compiles your template's keyword sets (modality, condition, "
               "constraint markers) into regular expressions and narrows the "
               "sentence in six stages: leading condition → modality + subject "
               "→ trailing condition → constraint → function type → "
               "action/object split.",
        "best": "Boilerplate IREB/Rupp & EARS requirements; CI pipelines and "
                "audit trails where repeatable, explainable output matters.",
        "needs": "No dependencies — pure Python. ~0.05 ms per requirement.",
    },
    "spacy": {
        "title": "spaCy · dependency parse",
        "how": "Runs the spaCy NLP pipeline to build a grammatical dependency "
               "tree, then maps relations to roles: advcl → CONDITION, nsubj → "
               "SUBJECT, modal aux → MODALITY, dobj → OBJECT, coordinated verbs "
               "→ multiple PROCESS. Falls back to rules if no ROOT verb is found.",
        "best": "Complex, nested, or free-prose sentences that don't follow a "
                "fixed boilerplate.",
        "needs": "pip install spacy + a model (en_core_web_sm). ~5–15 ms/req.",
    },
    "bert": {
        "title": "BERT · fine-tuned token tagger",
        "how": "A transformer token classifier assigns an IREB role (BIO scheme) "
               "to every word-piece, trained on labelled requirements. Character "
               "offsets are reconstructed from the predicted tags.",
        "best": "Domain-specific vocabulary where you have labelled data and the "
                "other backends mislabel the same patterns.",
        "needs": "pip install torch transformers + a trained model directory. "
                 "~10–30 ms/req on CPU.",
    },
}

TEMPLATE_INFO = {
    "IREB-Rupp": "IREB / Rupp 'MASTeR' boilerplate: <CONDITION>, the <SUBJECT> "
                 "<MODALITY> [provide <ACTOR> with the ability to] <PROCESS> "
                 "<OBJECT> <CONSTRAINT>. The most expressive shipped structure.",
    "EARS": "Easy Approach to Requirements Syntax (common in aerospace): "
            "When/While/If/Where <trigger>, the <SUBJECT> shall <PROCESS> "
            "<OBJECT>. Leaner marker set than Rupp.",
}


def template_info(name: str) -> str:
    """Describe a template, deriving a generic blurb for custom ones."""
    if name in TEMPLATE_INFO:
        return TEMPLATE_INFO[name]
    t = TEMPLATES.get(name)
    if t is None:
        return ""
    mods = ", ".join(t.modality_keywords[:4])
    return (f"Custom template. Modality keywords: {mods}…  "
            f"Slots: {' → '.join(r.value for r in t.slot_order)}.")


class GuiState:
    """Shared, thread-safe backend registry for the GUI server."""

    def __init__(self, bert_model_dir: Optional[str] = None):
        self.bert_model_dir = bert_model_dir or DEFAULT_BERT_DIR
        self._lock = threading.Lock()
        self._extractors = {"rules": RuleExtractor()}

    def backends(self) -> dict:
        avail = {"rules": True, "spacy": False,
                 "bert": os.path.isdir(self.bert_model_dir)}
        try:
            avail["spacy"] = SpacyExtractor().available()
        except Exception:
            pass
        return avail

    def extractor(self, name: str):
        with self._lock:
            if name not in self._extractors:
                if name == "spacy":
                    self._extractors[name] = SpacyExtractor()
                elif name == "bert":
                    self._extractors[name] = BertTaggerExtractor(
                        model_dir=self.bert_model_dir)
                else:
                    raise ReqGraphError(f"unknown backend {name!r}")
            return self._extractors[name]


# ---------------------------------------------------------------------------
# payload builders (pure functions, unit-testable)
# ---------------------------------------------------------------------------

def _graph_to_tree(g) -> Optional[dict]:
    """Nested semantic tree (ROOT down), children sorted in text order."""
    if not g.root_id:
        return None
    pos = {nid: i for i, nid in enumerate(g.leaf_order)}
    children: dict = {}
    for e in g.edges:
        if e.rel is not Rel.NEXT:
            children.setdefault(e.source, []).append(e)

    def min_leaf(nid):
        if nid in pos:
            return pos[nid]
        kids = children.get(nid, [])
        return min((min_leaf(e.target) for e in kids), default=10 ** 9)

    def build(nid):
        n = g.nodes[nid]
        kids = sorted(children.get(nid, []), key=lambda e: min_leaf(e.target))
        return {"id": n.id, "role": n.role.value, "text": n.text.strip(),
                "attrs": n.attrs,
                "children": [{"rel": e.rel.value, "node": build(e.target)}
                             for e in kids]}

    return build(g.root_id)


def _compute_kpis(g, text: str, parse_ms: float) -> dict:
    elements = g.elements()
    sem_chars = sum(len(n.text) for n in elements)
    q = g.analysis.get("quality", {})
    smells = [label for key, label in (
        ("missing_modality", "missing modality"),
        ("passive_voice", "passive voice"),
        ("vague_quantifier", "vague quantifier"),
        ("non_atomic", "non-atomic"),
        ("compound_requirement", "compound requirement"))
        if q.get(key)]
    weak = list(q.get("weak_words", []))
    score = max(0, 100 - 20 * len(smells) - 10 * len(weak))
    smells += [f"weak word: {w}" for w in weak]
    modality = next((n for n in elements if n.role is Role.MODALITY), None)
    n_actions = sum(1 for n in g.nodes.values() if n.role is Role.ACTION)
    op = next((n for n in g.nodes.values() if n.role is Role.OPERATOR), None)
    return {
        "roundtrip_ok": g.generate() == text,
        "parse_ms": round(parse_ms, 2),
        "n_elements": len(elements),
        "coverage_pct": round(100.0 * sem_chars / len(text), 1) if text else 0.0,
        "quality_score": score,
        "smells": smells,
        "type": g.analysis.get("type", ""),
        "ears_pattern": g.analysis.get("ears_pattern", ""),
        "obligation": (modality.attrs.get("obligation", "") if modality else "none"),
        "n_actions": n_actions,
        "operator": (op.attrs.get("operator") if op else None),
        "n_words": len(text.split()),
    }


def _requirement_payload(parser: RequirementParser, text: str) -> dict:
    t0 = time.perf_counter()
    g = parser.parse(text)
    parse_ms = (time.perf_counter() - t0) * 1000.0
    enrich(g)
    return {
        "text": text,
        "kpis": _compute_kpis(g, text, parse_ms),
        "tiles": [{"role": g.nodes[nid].role.value, "text": g.nodes[nid].text}
                  for nid in g.leaf_order],
        "tree": _graph_to_tree(g),
        "elements": [{"role": n.role.value, "text": n.text.strip(),
                      "attrs": n.attrs} for n in g.elements()],
        "mermaid": g.to_mermaid(),
        "dot": g.to_dot(),
        "cypher": g.to_cypher(),
        "graphml": g.to_graphml(),
        "turtle": g.to_turtle(),
        "graph": g.to_dict(),
    }


def parse_request(state: GuiState, payload: dict) -> dict:
    """Handle one /api/parse payload; returns the full UI model as a dict.

    A compound input ("...shall X and ...shall Y") is split into independent
    requirements, each parsed separately and listed under ``requirements``.
    The top-level fields mirror the first requirement for backward
    compatibility (identical to the old shape for single-requirement input).
    """
    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise ReqGraphError("please enter a requirement text")
    template = TEMPLATES.get(payload.get("template", ""), RUPP_TEMPLATE)
    backend = payload.get("backend", "rules")
    parser = RequirementParser(template, state.extractor(backend))

    segments = parser.split(text) if payload.get("split", True) else [text]
    requirements = [_requirement_payload(parser, seg) for seg in segments]

    out = dict(requirements[0])
    out.update({
        "requirements": requirements,
        "n_requirements": len(requirements),
        "template": template.name,
        "backend": backend,
    })
    return out


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    state: GuiState = None  # injected by launch()

    # quiet: route access logs to the logger instead of stderr
    def log_message(self, fmt, *args):
        logger.debug("gui: " + fmt, *args)

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                body = (_STATIC_DIR / "index.html").read_bytes()
            except OSError:
                self.send_error(500, "static/index.html missing")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/info":
            self._send_json({"templates": sorted(TEMPLATES),
                             "backends": self.state.backends(),
                             "backend_info": BACKEND_INFO,
                             "template_info": {n: template_info(n)
                                               for n in sorted(TEMPLATES)},
                             "version": __version__})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/parse":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._send_json(parse_request(self.state, payload))
        except ReqGraphError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:  # never leak a traceback to the page
            logger.exception("gui parse failed")
            self._send_json({"error": f"internal error: {exc}"}, status=500)


def make_server(port: int = 8765, bert_model_dir: Optional[str] = None):
    """Build (but don't start) the GUI server; port 0 picks a free port."""
    handler = type("BoundHandler", (_Handler,),
                   {"state": GuiState(bert_model_dir)})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def launch(port: int = 8765, open_browser: bool = True,
           bert_model_dir: Optional[str] = None) -> int:
    server = make_server(port, bert_model_dir)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"reqgraph GUI running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        server.server_close()
    return 0
