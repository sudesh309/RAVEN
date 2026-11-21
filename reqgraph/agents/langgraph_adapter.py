"""
LangGraph wiring for the review agents.

    from reqgraph.agents.langgraph_adapter import build_review_graph
    graph = build_review_graph(llm=get_provider("ollama"))
    out = graph.invoke(initial_state(items, load_architecture("model.sysml")))
    out["report"]["summary"]

Topology::

    START → planner ─┬→ writing_quality ─────────────┐
                     ├→ completeness ────────────────┤
                     ├→ rationale ───────────────────┤
                     ├→ consistency ─────────────────┼→ synthesis → END
                     ├→ cascade            (if links/levels)
                     ├→ allocation         (if model) ┤
                     ├→ architecture_justification ──┤
                     └→ derived_requirements ────────┘

The planner fans out (in one super-step, so the agents run in parallel) only to
the agents that apply; ``results`` uses a dict-merge reducer so the parallel
partial updates combine. ``langgraph`` is an optional dependency
(``pip install langgraph``); everything else in :mod:`reqgraph.agents` works
without it.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from .base import ReviewContext, normalise_requirements
from .orchestrator import default_agents, synthesise


def _merge(a: dict, b: dict) -> dict:
    return {**(a or {}), **(b or {})}


class ReviewState(TypedDict, total=False):
    requirements: list
    architecture: Optional[dict]
    config: dict
    plan: list
    results: Annotated[dict, _merge]
    report: dict


def initial_state(requirements, architecture=None, config=None) -> dict:
    if architecture is not None and not isinstance(architecture, dict):
        architecture = architecture.to_dict()
    return {"requirements": normalise_requirements(requirements),
            "architecture": architecture, "config": dict(config or {}),
            "plan": [], "results": {}}


def build_review_graph(agents=None, *, llm=None, names=None, checkpointer=None):
    """Compile a LangGraph ``StateGraph`` running the review agents.

    ``agents`` — agent instances (default: all); ``names`` — or a subset by
    agent / team name; ``llm`` — provider handed to every agent.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:                       # pragma: no cover - optional
        raise ImportError("LangGraph is not installed: pip install langgraph") from exc

    agents = list(agents) if agents is not None else default_agents(llm, names)
    for a in agents:
        if a.llm is None:
            a.llm = llm

    def planner(state):
        ctx = ReviewContext.from_state(state)
        return {"plan": [a.name for a in agents if a.applicable(ctx)[0]]}

    def route(state):
        return state.get("plan") or ["synthesis"]

    def synthesis(state):
        return {"report": synthesise(state)}

    g = StateGraph(ReviewState)
    g.add_node("planner", planner)
    for a in agents:
        g.add_node(a.name, a)
        g.add_edge(a.name, "synthesis")
    g.add_node("synthesis", synthesis)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", route, [a.name for a in agents] + ["synthesis"])
    g.add_edge("synthesis", END)
    return g.compile(checkpointer=checkpointer) if checkpointer else g.compile()
