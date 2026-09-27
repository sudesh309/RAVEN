"""
reqgraph.llm
============

Pluggable LLM access for RAVEN's assistive features, with two back ends:

* **Ollama** — a local model server (``ollama serve``). Nothing leaves the
  machine, which is what an export-controlled programme needs.
* **Google Cloud Vertex AI** — Gemini models via the ``generateContent`` REST
  API, authenticated with Application Default Credentials, the ``gcloud`` CLI,
  or an explicit bearer token. The endpoint can be overridden for Private
  Service Connect, a sovereign-cloud host, or an internal gateway.

Both use only the standard library (``urllib``); ``google-auth`` is used when
installed (``pip install reqgraph[gcp]``) but is not required.

Governance
----------
An LLM is an assistant here, never an authority. Everything it returns is a
**candidate**: rewrite suggestions are re-scored by RAVEN's own deterministic
quality and completeness checks, and LLM-refined PlusCal is re-validated by TLC
(see :mod:`reqgraph.assist`). Nothing is applied without a human choosing to.

Configuration (environment, all optional)
-----------------------------------------
=========================  ====================================================
``RAVEN_LLM_PROVIDER``     ``ollama`` | ``vertex`` | ``none`` (default: none)
``OLLAMA_HOST``            default ``http://localhost:11434``
``RAVEN_OLLAMA_MODEL``     default ``llama3.1``
``GOOGLE_CLOUD_PROJECT``   GCP project id (also ``GCLOUD_PROJECT``)
``GOOGLE_CLOUD_LOCATION``  region, or ``global`` (default ``us-central1``)
``RAVEN_VERTEX_MODEL``     default ``gemini-2.5-flash``
``RAVEN_VERTEX_ENDPOINT``  base URL override (PSC / sovereign cloud / gateway)
``GOOGLE_OAUTH_ACCESS_TOKEN``  explicit bearer token (skips credential lookup)
=========================  ====================================================
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from .errors import ReqGraphError


class LLMError(ReqGraphError):
    """An LLM back end is unreachable, misconfigured, or returned no answer."""


_LOOPBACK = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _is_loopback(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in _LOOPBACK or host.startswith("127.")


def _http_json(url: str, payload=None, *, headers=None, timeout: float = 120,
               bypass_proxy: bool = False) -> dict:
    """POST (or GET when ``payload`` is None) JSON and decode the JSON reply."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 method="POST" if data is not None else "GET")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    # A corporate HTTP(S)_PROXY must not intercept calls to a local model server.
    handlers = [urllib.request.ProxyHandler({})] if bypass_proxy else []
    opener = urllib.request.build_opener(*handlers)
    try:
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:600].decode("utf-8", "replace")
        try:                                  # Google errors: {"error": {"message": …}}
            detail = json.loads(detail).get("error", {}).get("message", detail)
        except Exception:
            pass
        raise LLMError(f"HTTP {exc.code} from {_redact(url)}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"cannot reach {_redact(url)}: {exc.reason}") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise LLMError(f"no reply from {_redact(url)} within {timeout:.0f}s") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMError(f"non-JSON reply from {_redact(url)}: {body[:200]!r}") from exc


def _redact(url: str) -> str:
    """Strip query strings (which can carry API keys) from URLs in messages."""
    return url.split("?", 1)[0]


def extract_json(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating ``` fences and prose."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S)
    if fence:
        t = fence.group(1)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(t[start:end + 1])
            except json.JSONDecodeError:
                pass
    raise LLMError(f"model did not return valid JSON: {text[:200]!r}")


class LLMProvider:
    """Common interface: ``complete()`` for a reply, ``status()`` for a health check."""

    name = "base"
    model = ""

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 temperature: float = 0.2, json_mode: bool = False) -> str:
        raise NotImplementedError

    def status(self) -> dict:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}:{self.model}"


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------

def _normalise_ollama_host(host: str) -> str:
    """Accept the forms OLLAMA_HOST takes in practice ("0.0.0.0:11434", …)."""
    h = host.strip()
    if "://" not in h:
        h = "http://" + h
    parts = urllib.parse.urlparse(h)
    hostname = parts.hostname or "localhost"
    if hostname in ("0.0.0.0", "::"):           # a bind address, not a target
        hostname = "127.0.0.1"
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    port = parts.port or 11434
    return f"{parts.scheme}://{hostname}:{port}{parts.path.rstrip('/')}"


class OllamaProvider(LLMProvider):
    """A model served by a local (or LAN) Ollama instance."""

    name = "ollama"
    DEFAULT_MODEL = "llama3.1"

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None,
                 timeout: float = 300, **_ignored):
        self.host = _normalise_ollama_host(
            host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434")
        self.model = model or os.environ.get("RAVEN_OLLAMA_MODEL") or self.DEFAULT_MODEL
        self.timeout = timeout

    def complete(self, prompt, *, system=None, temperature=0.2, json_mode=False):
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"temperature": temperature}}
        if json_mode:
            payload["format"] = "json"
        out = _http_json(f"{self.host}/api/chat", payload, timeout=self.timeout,
                         bypass_proxy=_is_loopback(self.host))
        if out.get("error"):
            raise LLMError(f"Ollama: {out['error']}")
        content = (out.get("message") or {}).get("content")
        if content is None:
            raise LLMError(f"unexpected Ollama reply: {str(out)[:200]}")
        return content

    def _model_present(self, names) -> bool:
        m = self.model
        return any(n == m or (":" not in m and n.split(":")[0] == m) for n in names)

    def status(self) -> dict:
        base = {"provider": self.name, "model": self.model, "host": self.host}
        try:
            tags = _http_json(f"{self.host}/api/tags", timeout=5,
                              bypass_proxy=_is_loopback(self.host))
        except LLMError as exc:
            return {**base, "available": False, "reachable": False,
                    "detail": f"{exc}. Start the server with `ollama serve`."}
        names = [m.get("name", "") for m in tags.get("models", [])]
        present = self._model_present(names)
        return {**base, "available": present, "reachable": True, "models": names,
                "detail": ("ready" if present else
                           f"model '{self.model}' is not pulled -- run "
                           f"`ollama pull {self.model}`"
                           + (f" (installed: {', '.join(names[:6])})" if names else ""))}


# ---------------------------------------------------------------------------
# Google Cloud Vertex AI
# ---------------------------------------------------------------------------

class VertexAIProvider(LLMProvider):
    """Gemini on Vertex AI (``generateContent``)."""

    name = "vertex"
    DEFAULT_MODEL = "gemini-2.5-flash"
    _SCOPE = "https://www.googleapis.com/auth/cloud-platform"

    def __init__(self, model: Optional[str] = None, project: Optional[str] = None,
                 location: Optional[str] = None, access_token: Optional[str] = None,
                 endpoint: Optional[str] = None, timeout: float = 300, **_ignored):
        env = os.environ
        self.project = (project or env.get("GOOGLE_CLOUD_PROJECT")
                        or env.get("GCLOUD_PROJECT") or env.get("CLOUDSDK_CORE_PROJECT"))
        self.location = (location or env.get("GOOGLE_CLOUD_LOCATION")
                         or env.get("GOOGLE_CLOUD_REGION") or "us-central1")
        self.model = model or env.get("RAVEN_VERTEX_MODEL") or self.DEFAULT_MODEL
        self.endpoint = (endpoint or env.get("RAVEN_VERTEX_ENDPOINT") or "").rstrip("/")
        self.timeout = timeout
        self._token = access_token
        self._token_expiry = float("inf") if access_token else 0.0
        self._token_source = "explicit token" if access_token else None

    # -- endpoint -------------------------------------------------------------
    def base_url(self) -> str:
        if self.endpoint:
            return self.endpoint
        if self.location == "global":
            return "https://aiplatform.googleapis.com"
        return f"https://{self.location}-aiplatform.googleapis.com"

    def url(self) -> str:
        return (f"{self.base_url()}/v1/projects/{self.project}/locations/"
                f"{self.location}/publishers/google/models/{self.model}:generateContent")

    # -- credentials ----------------------------------------------------------
    def _access_token(self) -> str:
        if self._token and time.time() < self._token_expiry:
            return self._token
        tok = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
        if tok:
            self._token, self._token_expiry = tok, float("inf")
            self._token_source = "GOOGLE_OAUTH_ACCESS_TOKEN"
            return tok
        errors = []
        try:                                        # Application Default Credentials
            import google.auth
            import google.auth.transport.requests
            creds, proj = google.auth.default(scopes=[self._SCOPE])
            creds.refresh(google.auth.transport.requests.Request())
            if not self.project and proj:
                self.project = proj
            self._token, self._token_expiry = creds.token, time.time() + 45 * 60
            self._token_source = "application default credentials"
            return self._token
        except ImportError:
            errors.append("google-auth not installed")
        except Exception as exc:                    # no ADC configured, expired, …
            errors.append(f"ADC: {exc}")
        gcloud = shutil.which("gcloud")
        if gcloud:
            for args, src in ((["auth", "application-default", "print-access-token"],
                               "gcloud ADC"),
                              (["auth", "print-access-token"], "gcloud user")):
                try:
                    r = subprocess.run([gcloud, *args], capture_output=True,
                                       text=True, timeout=30)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    errors.append(f"{src}: {exc}")
                    continue
                if r.returncode == 0 and r.stdout.strip():
                    self._token, self._token_expiry = r.stdout.strip(), time.time() + 45 * 60
                    self._token_source = src
                    return self._token
                errors.append(f"{src}: {r.stderr.strip()[:120] or 'no token'}")
        else:
            errors.append("gcloud CLI not found")
        raise LLMError(
            "no Google Cloud credentials: run `gcloud auth application-default "
            "login`, or set GOOGLE_OAUTH_ACCESS_TOKEN, or install "
            f"`reqgraph[gcp]` with a service account ({'; '.join(errors)})")

    # -- API ------------------------------------------------------------------
    def complete(self, prompt, *, system=None, temperature=0.2, json_mode=False):
        token = self._access_token()
        if not self.project:
            raise LLMError("no GCP project: set GOOGLE_CLOUD_PROJECT or pass project=")
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                   "generationConfig": {"temperature": temperature}}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        out = _http_json(self.url(), payload, timeout=self.timeout,
                         headers={"Authorization": f"Bearer {token}"},
                         bypass_proxy=_is_loopback(self.base_url()))
        cands = out.get("candidates") or []
        if not cands:
            fb = out.get("promptFeedback") or {}
            raise LLMError("Vertex AI returned no answer"
                           + (f" (blocked: {fb.get('blockReason')})" if fb else ""))
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        if not text:
            raise LLMError(f"Vertex AI returned an empty answer "
                           f"(finishReason={cands[0].get('finishReason')})")
        return text

    def status(self) -> dict:
        """Check configuration and credentials without making a billable call."""
        base = {"provider": self.name, "model": self.model,
                "location": self.location, "endpoint": self.base_url()}
        try:
            self._access_token()
        except LLMError as exc:
            return {**base, "project": self.project, "available": False,
                    "detail": str(exc)}
        if not self.project:
            return {**base, "project": None, "available": False,
                    "detail": "credentials found, but no project: set "
                              "GOOGLE_CLOUD_PROJECT"}
        return {**base, "project": self.project, "available": True,
                "credentials": self._token_source,
                "detail": f"ready (credentials: {self._token_source})"}


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

PROVIDERS = {"ollama": OllamaProvider, "vertex": VertexAIProvider,
             "vertexai": VertexAIProvider, "gcp": VertexAIProvider,
             "google": VertexAIProvider, "gemini": VertexAIProvider}
_OFF = {"", "none", "off", "disabled", "false"}


def get_provider(name: Optional[str] = None, **options) -> Optional[LLMProvider]:
    """Build a provider by name (or ``$RAVEN_LLM_PROVIDER``); None when disabled.

    Empty option values are ignored so a half-filled GUI form falls back to the
    environment defaults instead of overriding them with blanks.
    """
    key = (name if name is not None else os.environ.get("RAVEN_LLM_PROVIDER", ""))
    key = key.strip().lower()
    if key in _OFF:
        return None
    cls = PROVIDERS.get(key)
    if cls is None:
        raise LLMError(f"unknown LLM provider {name!r}; use 'ollama' or 'vertex'")
    return cls(**{k: v for k, v in options.items() if v not in (None, "")})


def all_status() -> dict:
    """Health of every back end with its environment defaults (for UIs / CLI)."""
    return {"configured": (os.environ.get("RAVEN_LLM_PROVIDER") or "none").lower(),
            "ollama": OllamaProvider().status(),
            "vertex": VertexAIProvider().status()}
