# RAVEN agentic review — architecture

The agentic review answers four questions about a requirement baseline:

1. **Is each requirement well written and complete?** It should have proper wording, a real
   rationale, a source and a verification method.
2. **Is the set consistent?** It should have no duplicates or contradictions.
3. **Does the cascade hold?** Each lower-level requirement should trace to a parent. Each
   parent's children should cover what it asks for, and no child budget should be looser
   than its parent's.
4. **When a SysML model exists, do requirements and architecture explain each other?**
   - Every requirement should be realised by an element.
   - Every element should be justified by a requirement.
   - Requirements written in the model should appear in the specification.
   - Every decomposition should carry its derived requirements.

## Agents and teams

```
                         ┌──────────────── planner (deterministic) ────────────────┐
                         │  quality: always · cascade: if links/levels ·            │
                         │  architecture: only if a model is supplied               │
                         └──────────────────────────────────────────────────────────┘
        QUALITY TEAM (set only)            CASCADE            ARCHITECTURE TEAM (needs model)
  ┌──────────────────────────────┐   ┌──────────────┐   ┌───────────────────────────────────┐
  │ writing_quality              │   │ cascade      │   │ allocation        req → arch      │
  │ completeness                 │   │              │   │ architecture_justification        │
  │ rationale (+ source, V&V)    │   │              │   │                   arch → req      │
  │ consistency                  │   │              │   │ derived_requirements              │
  └──────────────────────────────┘   └──────────────┘   └───────────────────────────────────┘
                         └────────────────────── synthesis ─────────────────────────┘
               per-requirement scorecard (ready / needs work / blocked), per-element list,
               severity-ranked findings → dict · Markdown · CSV
```

The quality and architecture teams are **separate on purpose**. Loading a model never changes
the quality verdict on the requirement set. The architecture team only adds findings about the
link between the two.

| Agent | Module | Main rules |
|---|---|---|
| `writing_quality` | `agents/quality_agents.py` | `missing_modality`, `weak_words`, `vague_quantifier`, `non_atomic`, `compound_requirement`, `passive_voice` |
| `completeness` | 〃 | per requirement: `no_subject`, `placeholder`, `number_without_unit`, `no_tolerance`, … Per set: `no_<type>_requirements`, `no_off_nominal` |
| `rationale` | 〃 | `missing_rationale` (blocker on safety requirements), `trivial_rationale`, `restating_rationale`, `rationale_without_reason`, `unjustified_value`, `missing_source`, `missing_verification_method`, `unknown_verification_method`, `weak_verification_method` |
| `consistency` | 〃 | `duplicate_id`, `duplicate`, `polarity_conflict`, `value_conflict`, `similar` |
| `cascade` | `agents/cascade.py` | `dangling_parent`, `cycle`, `orphan`, `level_inversion`, `level_skip`, `not_decomposed`, `weak_flowdown`, `restated_child`, `budget_violation` |
| `allocation` | `agents/architecture_agents.py` | `unallocated`, `implicit_allocation`, `unknown_allocation_target` |
| `architecture_justification` | 〃 | `unjustified_element`, `unspecified_interface`, `unspecified_function` |
| `derived_requirements` | 〃 | `model_requirement_missing`, `derived_requirement_expected`, `allocation_outside_parent_branch`, `level_depth_mismatch` |

### How a requirement is linked to the architecture

The `allocation` agent links each requirement to architecture elements in three steps, strongest
evidence first:

1. **Same requirement in the model.** The agent looks for a SysML `requirement <'ID'>` whose
   short id or name equals the requirement's id. If none matches, it accepts a model
   requirement whose text is at least 75 % similar.
2. **Explicit links.** These come from:
   - the set's `allocated_to` / `satisfies` / `component` columns;
   - the model's `satisfy R by p;` statements, including `satisfy R;` written inside a part;
   - `allocate R to p;` statements;
   - the requirement's `subject`.

   Each reference is resolved to its definition. For example, the usage `drive : DriveUnit`
   resolves to `DriveUnit`.
3. **Semantic match (a candidate only).** The requirement's grammatical subject, as parsed by
   RAVEN, is compared with element names split at camel-case boundaries and with usage names.

Only steps 1 and 2 count as confirmed. A step-3 match is reported as `implicit_allocation`, with
a hint to make the link explicit.

## Agent contract

Every agent subclasses `ReviewAgent` (`agents/base.py`):

```python
class MyAgent(ReviewAgent):
    name = "my_agent"; domain = "quality" | "cascade" | "architecture"
    requires_architecture = False
    llm_instruction = "…"            # optional: enables the LLM pass

    def applicable(self, ctx) -> (bool, reason)      # planner uses this
    def analyze(self, ctx, result)                   # deterministic; result.add(rule, severity, msg, subject)
    def llm_payload(self, ctx, result) -> list       # optional: what the LLM reviews
```

- **LangGraph node.** `agent(state) -> {"results": {name: AgentResult-dict}}`. The state holds
  only JSON: requirements are `{id, text, meta}` dicts and the architecture is
  `ArchitectureModel.to_dict()`. This means the state can be checkpointed, sent over MCP, or
  logged.
- **Finding.** Fields are `{agent, rule, severity: blocker|major|minor|info, subject_id,
  subject_kind: requirement|element|set, message, hint, evidence, source:
  deterministic|llm, status: confirmed|candidate}`.
- **Failure isolation.** If an agent raises, its result has `status: "error"` and the other
  agents still report.

### The LLM step (optional)

When the orchestrator is given a provider (Ollama, Vertex AI, or any object with
`complete(prompt, system=, temperature=, json_mode=)`), each agent that sets `llm_instruction`
sends batches of its items. It also sends the rules it has already raised, so the LLM looks for
what the rules miss. Replies are JSON findings, guarded as follows:

- Every LLM finding is `source="llm"` and `status="candidate"`.
- An LLM finding is never more severe than `major`, so an LLM cannot block a baseline.
- Findings about subjects that are not in the set or the model are dropped as hallucinations.
- Calls are capped per agent (`llm_max_calls`).
- If the provider fails, the error is recorded in `data.llm_errors` and the rule-based result
  stands.

| Agent | What the LLM is asked |
|---|---|
| `writing_quality` | ambiguity a rule misses: undefined terms, implicit assumptions, design-as-requirement |
| `completeness` | missing modes and conditions, failure behaviour, undefined interfaces |
| `rationale` | does the rationale actually justify the requirement and its values? |
| `consistency` | logical conflicts, inconsistent terminology across the set |
| `cascade` | are this parent's children *sufficient* to satisfy it, and which aspect is not refined? |
| `allocation` | which element should satisfy an unallocated requirement? |
| `architecture_justification` | is an unjustified element gold-plating or a missing requirement? Draft the requirement |
| `derived_requirements` | draft the derived requirement a decomposed element needs |

## Runtimes

| Runtime | Entry point |
|---|---|
| Python | `reqgraph.agents.review(items, architecture, llm=…)` or `ReviewOrchestrator` (thread-pool fan-out) |
| LangGraph | `reqgraph.agents.langgraph_adapter.build_review_graph(llm=…)`: `START → planner → (fan-out to planned agents) → synthesis → END`. `results` uses a dict-merge reducer |
| Tools | `reqgraph.agents.tools.TOOL_SPECS` (JSON Schema) and `call_tool(name, args)`; `as_langchain_tools()` for LangChain / `ToolNode` |
| MCP | `python -m reqgraph mcp-server` (stdio, JSON-RPC 2.0, `tools/list` and `tools/call`) |
| CLI | `python -m reqgraph review reqs.reqif -a model.sysml [--llm --provider ollama] [--fail-on blocker]` |
| GUI | card 7, *Agentic review* (`POST /api/review`) |
| Agent skill | `skills/raven-requirements-review/SKILL.md` |

Example MCP configuration for Claude Code or Claude Desktop:

```json
{"mcpServers": {"raven": {"command": "python", "args": ["-m", "reqgraph", "mcp-server"]}}}
```

### Adding an agent to a LangGraph application

```python
from langgraph.graph import StateGraph
from reqgraph.agents import default_agents
from reqgraph.agents.langgraph_adapter import ReviewState, build_review_graph

graph = build_review_graph(agents=default_agents(llm) + [MyAgent(llm=llm)])
# or embed a single agent as a node in your own graph:
g = StateGraph(ReviewState); g.add_node("rationale", RationaleAgent(llm=llm))
```

## Architecture input

`reqgraph/architecture.py` reads two kinds of model:

- **SysML v2 textual notation.** It extracts:
  - part, port, action, state, interface and item definitions and usages, with their `doc`
    text;
  - the composition tree;
  - requirements, with short ids, `doc` or `text`, `subject` and nesting;
  - `satisfy … by …`, `allocate … to …`, `verify`, `dependency`, and `#derivation` connections.
- **SysML v1 (XMI, Turtle, Cameo `.mdzip`).** These are adapted from `SysMLV1Model`.

Requirements are read from ReqIF, CSV, Excel, JSON, Word and TXT. ReqIF `SPEC-RELATION`s become
`parent` links. The level column decides direction when levels are present; otherwise the
relation's SOURCE is taken as the parent.

## Limits

- **The SysML v2 reader uses patterns.** It covers common textual constructs, not the full KerML
  grammar. Anything it cannot place is left out; it never invents links.
- **Term coverage is not semantic sufficiency.** `weak_flowdown` compares words, so it is a
  *candidate* finding. A parent's children can refine it in different vocabulary. Use the LLM
  step, or engineering review, to judge sufficiency.
- **Budget checks are narrow.** `budget_violation` compares only like units with matching
  context words and the same bound direction (upper or lower).
- **Implicit allocation is a candidate.** It is a name match. A baseline review should make
  every allocation explicit.
