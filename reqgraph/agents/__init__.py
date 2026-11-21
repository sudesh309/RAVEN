"""
reqgraph.agents — multi-agent requirement review.

Teams (see ``docs/AGENTIC_ARCHITECTURE.md``):

* **Quality** (requirement set only): ``writing_quality``, ``completeness``,
  ``rationale``, ``consistency``.
* **Cascade** (set hierarchy): ``cascade``.
* **Architecture** (needs a SysML model): ``allocation``,
  ``architecture_justification``, ``derived_requirements``.

Every agent is a LangGraph-compatible node (``agent(state) -> partial update``);
:class:`ReviewOrchestrator` runs them without LangGraph,
:func:`~reqgraph.agents.langgraph_adapter.build_review_graph` builds a LangGraph
``StateGraph``, :mod:`reqgraph.agents.tools` exposes them as tool specs and
:mod:`reqgraph.agents.mcp_server` serves them over MCP.
"""

from .base import (SEVERITIES, AgentResult, Finding, ReviewAgent, ReviewContext,
                   normalise_requirements)
from .quality_agents import (CompletenessAgent, ConsistencyAgent, RationaleAgent,
                             WritingQualityAgent)
from .cascade import CascadeAgent
from .architecture_agents import (AllocationAgent, ArchitectureJustificationAgent,
                                  DerivedRequirementsAgent)
from .orchestrator import (AGENT_CLASSES, TEAMS, ReviewOrchestrator, ReviewReport,
                           default_agents, plan, review, synthesise)

__all__ = ["SEVERITIES", "AgentResult", "Finding", "ReviewAgent", "ReviewContext",
           "normalise_requirements", "WritingQualityAgent", "CompletenessAgent",
           "RationaleAgent", "ConsistencyAgent", "CascadeAgent", "AllocationAgent",
           "ArchitectureJustificationAgent", "DerivedRequirementsAgent", "AGENT_CLASSES",
           "TEAMS", "ReviewOrchestrator", "ReviewReport", "default_agents", "plan",
           "review", "synthesise"]
