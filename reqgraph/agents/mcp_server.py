"""
Minimal MCP (Model Context Protocol) server over stdio — standard library only.

Exposes :data:`reqgraph.agents.tools.TOOL_SPECS` so any MCP host (Claude
Desktop / Claude Code, LangGraph via ``langchain-mcp-adapters``, IDE agents)
can call the RAVEN review agents::

    python -m reqgraph mcp-server

Claude Code / Desktop configuration::

    {"mcpServers": {"raven": {"command": "python",
                              "args": ["-m", "reqgraph", "mcp-server"]}}}

Messages are newline-delimited JSON-RPC 2.0 (the MCP stdio transport).
Implemented methods: ``initialize``, ``ping``, ``tools/list``, ``tools/call``;
notifications are accepted and ignored.
"""

from __future__ import annotations

import json
import sys
import traceback

from .tools import TOOL_SPECS, call_tool

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "raven-requirements-review", "version": "1.0"}


def handle(msg: dict, llm=None):
    """Handle one JSON-RPC message; returns the response dict or None."""
    mid = msg.get("id")
    method = msg.get("method", "")
    if mid is None:                       # notification (e.g. notifications/initialized)
        return None

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def err(code, message):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}

    if method == "initialize":
        requested = (msg.get("params") or {}).get("protocolVersion") or PROTOCOL_VERSION
        return ok({"protocolVersion": requested,
                   "capabilities": {"tools": {"listChanged": False}},
                   "serverInfo": SERVER_INFO,
                   "instructions": ("RAVEN requirement review agents. Call raven_review "
                                    "with a requirement set (and optionally a SysML model) "
                                    "for a full multi-agent review.")})
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": [{"name": t["name"], "description": t["description"],
                              "inputSchema": t["input_schema"]} for t in TOOL_SPECS]})
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name", "")
        if name not in {t["name"] for t in TOOL_SPECS}:
            return err(-32602, f"unknown tool: {name}")
        try:
            out = call_tool(name, params.get("arguments") or {}, llm=llm)
        except Exception as exc:          # tool errors are results, not protocol errors
            return ok({"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                       "isError": True})
        text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)
        result = {"content": [{"type": "text", "text": text}], "isError": False}
        if isinstance(out, dict):
            result["structuredContent"] = out
        return ok(result)
    return err(-32601, f"method not found: {method}")


def serve(stdin=None, stdout=None, llm=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": "parse error"}}
        else:
            try:
                resp = handle(msg, llm=llm)
            except Exception:             # never let one bad message kill the server
                traceback.print_exc(file=sys.stderr)
                resp = {"jsonrpc": "2.0", "id": msg.get("id"),
                        "error": {"code": -32603, "message": "internal error"}}
        if resp is not None:
            stdout.write(json.dumps(resp, ensure_ascii=False, default=str) + "\n")
            stdout.flush()


if __name__ == "__main__":                # pragma: no cover
    serve()
