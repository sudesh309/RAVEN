# RAVEN — Architecture Requirements & Decisions (ARD)

**Product:** RAVEN, the Requirement Analysis and Visualisation Engine (Python package `reqgraph`)
**Status:** describes the system as built on `main`
**Audience:** architects, tool owners, and engineers adopting or extending RAVEN

Related documents: `README.md` (use and configuration), `reqgraph/docs/ARCHITECTURE.md` (core
graph model), `reqgraph/docs/AGENTIC_ARCHITECTURE.md` (review agents), `reqgraph/static/roadmap.html`
(constraints and future plans).

---

## 1. Purpose and scope

### 1.1 Problem
Requirement sets are written as free text in Word, Excel and ReqIF. Quality problems (ambiguity,
missing rationale, broken decomposition, no link to the architecture) are found late, by hand, and
inconsistently. Architects need to know whether the requirements are good and whether the
architecture actually explains them.

### 1.2 What RAVEN does
1. Parses each requirement into a semantic graph (IREB roles: subject, modality, process, object,
   condition, constraint) and regenerates the text from the graph without loss.
2. Scores wording quality and completeness against INCOSE rules.
3. Finds connections, duplicates and conflicts across a set.
4. Imports Word, JSON, CSV, Excel and ReqIF, and checks traceability inside the document.
5. Compares requirements with SysML v2 or v1 models and builds a requirements verification and
   traceability matrix (RVTM).
6. Formalises requirements as PlusCal / TLA+ and model-checks them with TLC.
7. Runs a team of review agents over the set and, when a model is supplied, over the architecture.
8. Offers an optional local or cloud LLM that only ever *proposes*.

### 1.3 Out of scope
- Authoring or editing requirements in place, or acting as a requirements database.
- Multi-user collaboration, access control, or storage of baselines.
- Full SysML v2 / KerML grammar support.
- Proving that a rewrite or a candidate finding preserves meaning. A human confirms.

---

## 2. Stakeholders and use cases

| Stakeholder | Need | Where RAVEN serves it |
|---|---|---|
| Requirements engineer | Find weak, incomplete or conflicting requirements before review | Cards 1–3, review agents |
| System / aircraft architect | Check requirements cascade and that architecture and requirements explain each other | Cards 4–5, 7 |
| Safety / V&V engineer | Verification method for every requirement; formalise critical ones | RVTM, PlusCal |
| Programme / quality lead | Baseline readiness gate in CI | `review --fail-on` |
| Tool integrator | Embed checks in an agent platform | LangGraph, MCP, tool specs |

---

## 3. Requirements on the architecture

Identifiers: **FR** functional, **NFR** non-functional, **CON** constraint.

### 3.1 Functional
| ID | Requirement |
|---|---|
| FR-01 | Parse one requirement into a graph whose regenerated text equals the input (lossless round trip). |
| FR-02 | Report wording quality (modality, weak words, vague quantifiers, passive voice, atomicity) and completeness (subject, action, units, tolerance, TBD). |
| FR-03 | Detect connections, duplicates and conflicts across a set. |
| FR-04 | Import requirements from Word, JSON, CSV, Excel and ReqIF, keeping extra columns as metadata. |
| FR-05 | Check document-level traceability: identifiers, parent / satisfy / verify links, dangling and untraced items. |
| FR-06 | Compare requirements with a SysML v2 model, and with a SysML v1 model (XMI, Turtle, Cameo `.mdzip`), and produce an RVTM with verified / candidate / gap status. |
| FR-07 | Generate PlusCal / TLA+ from a requirement and validate it with TLC when available. |
| FR-08 | Review the set with separate agent teams: quality, cascade, and architecture. |
| FR-09 | The architecture team runs only when a model is supplied. |
| FR-10 | Optionally use a local (Ollama) or cloud (Vertex AI) LLM for rewrites, PlusCal refinement, and extra review findings. |
| FR-11 | Provide the same functions from a browser GUI, a CLI, a Python API, an MCP server, and LangGraph / LangChain tools. |

### 3.2 Non-functional
| ID | Requirement | How it is met |
|---|---|---|
| NFR-01 | Runs offline with no mandatory network or heavy dependency | Rule-based extraction and lexical similarity are the defaults; spaCy, BERT, pandas, lxml, LLM clients are optional extras |
| NFR-02 | Data stays on the user's machine unless they choose a cloud model | GUI binds to `127.0.0.1`; Ollama is local; Vertex AI is opt-in |
| NFR-03 | Results are reproducible and explainable | Deterministic rules produce all confirmed findings; each finding carries a rule key, evidence and hint |
| NFR-04 | An LLM can never silently change a verdict | LLM output is a *candidate*, capped in severity, checked against known ids, and re-scored or re-verified by RAVEN |
| NFR-05 | One component failing does not lose the rest | An agent that raises becomes `status: error`; others report |
| NFR-06 | Interactive speed on typical sets | A 40-requirement set with a model reviews in about 1 s; pairwise checks are bounded |
| NFR-07 | Safe local web surface | JSON-only POSTs, same-origin and Host checks, fixed page whitelist, no traceback leaks |
| NFR-08 | Extensible | New agent = one subclass; new input format = one reader; new LLM back end = one provider class |
| NFR-09 | Easy start on Windows and Linux | `pip install`, `python -m reqgraph gui`, Windows launcher |

### 3.3 Constraints
| ID | Constraint |
|---|---|
| CON-01 | Python 3.10+; core package has no required third-party dependency. |
| CON-02 | SysML v2 is read from textual notation by pattern matching, not a full parser. |
| CON-03 | Model checking needs a local Java and the TLA+ tools. |
| CON-04 | Vertex AI needs Google credentials on the host (ADC, `gcloud`, or token). |
| CON-05 | State is per session; nothing is persisted by the tool. |

---

## 4. Architecture overview

### 4.1 Layers

```
 Interfaces   GUI (static HTML + JSON API) · CLI · Python API · MCP server · LangGraph / LangChain
 ───────────────────────────────────────────────────────────────────────────────────────────────
 Review       agents/: orchestrator · quality team · cascade · architecture team · synthesis
 ───────────────────────────────────────────────────────────────────────────────────────────────
 Analysis     quality · traceability · corpus (set graph, similarity) · sysml_compare ·
              sysml_v1_compare · pluscal · assist (LLM-guarded rewrite / refine)
 ───────────────────────────────────────────────────────────────────────────────────────────────
 Model        core (graph, roles, relations) · parser · templates · tiling · extractors
 ───────────────────────────────────────────────────────────────────────────────────────────────
 Input        io_formats (Word, JSON, CSV, Excel, ReqIF) · sysml_parser · sysml_v1_parser ·
              architecture (SysML structure, links) · llm (Ollama, Vertex AI)
```

Dependencies point downward only. The interfaces share the same functions; none contains
analysis logic.

### 4.2 Key components

| Component | Responsibility |
|---|---|
| `core`, `parser`, `tiling`, `extractors`, `templates` | The requirement graph. Extractors (rules, spaCy, BERT) propose element spans; tiling guarantees every character belongs to exactly one node, which is what makes the round trip lossless. |
| `quality` | Wording smells, completeness findings with severities, requirement type and EARS pattern. |
| `corpus` | Set-level graph: connections, duplicates, conflicts; lexical or embedding similarity. |
| `io_formats` | File readers into a common `(id, text, metadata)` triple. ReqIF relations become `parent` links. |
| `traceability` | Verification-method suggestion, RVTM, document-level link checks. |
| `sysml_parser`, `sysml_compare`, `sysml_v1_*` | Model element extraction and role-bucketed similarity to requirements. |
| `architecture` | Structure for the architecture agents: composition tree, embedded requirements, satisfy / allocate / verify / derive links. Adapter for SysML v1. |
| `agents` | Eight review agents, a supervisor, synthesis, and connectors (see below). |
| `llm`, `assist` | Provider abstraction; guarded rewrite and PlusCal refinement. |
| `pluscal` | Requirement-to-PlusCal translation and TLC validation. |
| `gui`, `static/` | Local HTTP server and single-page UI. |

### 4.3 Review agent design

```
 planner ─┬→ writing_quality ─┐        quality team (set only)
          ├→ completeness ────┤
          ├→ rationale ───────┤
          ├→ consistency ─────┤
          ├→ cascade ─────────┼→ synthesis → report (dict · Markdown · CSV)
          ├→ allocation ──────┤        architecture team
          ├→ architecture_justification   (only if a model exists)
          └→ derived_requirements ┘
```

- Each agent is a function `state → {"results": {name: result}}`, so it is a LangGraph node as is.
- State is JSON only: requirements as `{id, text, meta}`, the architecture as a dict.
- The planner is deterministic. No LLM decides what gets checked.
- Synthesis turns findings into a per-requirement status (ready / needs work / blocked) and a
  per-element list.

### 4.4 Requirement-to-architecture linkage

The `allocation` agent links a requirement to architecture elements, strongest evidence first:
1. the same requirement written in the model (matching short id or name, else ≥ 75 % text match);
2. explicit links: `allocated_to` columns, `satisfy … by …`, `allocate … to …`, requirement
   `subject`;
3. a name match between the requirement's grammatical subject and element names. This is a
   *candidate* only.

Findings then run both ways: requirements without an element, and elements without a requirement.
Derived-requirement checks look at decomposition (a part broken into sub-parts should carry
requirements for them) and at the cascade (a child requirement should be allocated inside its
parent's architecture branch).

---

## 5. Architecture decisions

Each decision lists the context, the choice, the reasons, and what it costs.

### AD-01 — Graph model with lossless round trip
- **Context:** Requirement text must be analysed and edited without drift.
- **Decision:** Represent each requirement as a typed graph; tile the text so every character is in
  exactly one node; generate text back from the graph.
- **Why:** Analysis, comparison and export all work from one structure, and a lossy parse is
  detectable.
- **Cost:** Extraction errors show up as odd tiles rather than hidden. Rules do not cover every
  phrasing.

### AD-02 — Pluggable extraction backends, rules by default
- **Decision:** Rule extractor is the default; spaCy and a fine-tuned BERT tagger are optional.
- **Why:** NFR-01. Works anywhere, instantly, with no model download.
- **Cost:** Lower recall on unusual sentences than a trained model.

### AD-03 — Deterministic first, LLM second
- **Decision:** All confirmed findings come from rules. An LLM only adds findings marked
  `candidate`, never above `major`, only about real ids, with a call cap.
- **Why:** NFR-03 and NFR-04. Reviewers can trust and reproduce the base result; AI help cannot
  quietly block or clear a baseline.
- **Cost:** The LLM cannot override a wrong rule; a human must confirm candidates.

### AD-04 — Separate quality and architecture teams
- **Decision:** The requirement-quality agents never read the model. A distinct team handles the
  link between requirements and architecture, and runs only if a model is supplied.
- **Why:** The quality verdict must not change because a model happens to be loaded, and each
  team can be run, tested and gated on its own.
- **Cost:** Some checks use similar helpers in both teams.

### AD-05 — Agents as plain state-to-update functions
- **Decision:** Agents are callables on a JSON state, not tied to a framework. A LangGraph builder,
  LangChain tools and an MCP server wrap the same objects.
- **Why:** FR-11, NFR-08. The tool can join any agent platform, and runs without one.
- **Cost:** LangGraph features (human-in-the-loop interrupts, persistence) come from the host
  application, not from RAVEN.

### AD-06 — Stdlib-only MCP server and HTTP GUI
- **Decision:** Implement MCP stdio JSON-RPC and the GUI server with the standard library.
- **Why:** NFR-01, NFR-09. No framework to install or version.
- **Cost:** Only the MCP methods needed are implemented (initialize, ping, tools/list,
  tools/call); no streaming or resources.

### AD-07 — Own SysML reader instead of a full parser
- **Decision:** Pattern-based extraction of the common SysML v2 text constructs, and an XMI /
  Turtle reader for v1.
- **Why:** There was no stable, lightweight Python SysML v2 parser; the checks need only
  structure, docs, requirements and links.
- **Cost:** CON-02. Unusual syntax can be missed; the reader leaves out what it cannot place and
  never invents links.

### AD-08 — Common input triple and metadata pass-through
- **Decision:** Every reader returns `(id, text, metadata)`; unknown columns are kept.
- **Why:** Rationale, level, parent, verification method and allocation columns drive the
  agents without a fixed schema.
- **Cost:** Column naming is matched flexibly by alias lists, which can mis-map unusual headers.

### AD-09 — Local-first, same-origin web surface
- **Decision:** GUI listens on loopback only; POSTs must be JSON with matching Host and Origin;
  credentials for cloud models are never typed into the page.
- **Why:** NFR-02, NFR-07. A web page the user visits must not be able to drive the tool or use
  their cloud credentials.
- **Cost:** Not suitable as a shared server without a different deployment design.

### AD-10 — Findings as the common currency
- **Decision:** Every agent emits the same finding record (rule, severity, subject, message, hint,
  evidence, source, status).
- **Why:** One filterable list, one CSV, one CI gate (`--fail-on`), and a consistent per-
  requirement status.
- **Cost:** Agent-specific data goes into `evidence` or the agent's `data` block.

---

## 6. Interfaces

| Interface | Contract |
|---|---|
| CLI | `reqgraph parse \| batch \| train \| gui \| analyze \| export \| compare \| compare-v1 \| connections \| pluscal \| llm-status \| rewrite \| review \| mcp-server` |
| HTTP (local) | `POST /api/parse`, `/connections`, `/export`, `/compare`, `/compare-v1`, `/pluscal`, `/pluscal/validate`, `/review`, `/review/sample`, `/llm/*`; `GET /api/info` |
| Python | `RequirementParser`, `check_quality`, `check_completeness`, `build_requirement_set_graph`, `compare_sysml`, `review`, `ReviewOrchestrator`, `load_architecture` |
| LangGraph | `build_review_graph(llm=…)`, `initial_state(items, model)` |
| Tools | `TOOL_SPECS` (JSON Schema), `call_tool`, `as_langchain_tools` |
| MCP | tools `raven_review`, `raven_quality_review`, `raven_cascade_review`, `raven_architecture_review`, `raven_parse_architecture`, `raven_list_agents` |

---

## 7. Quality attributes and verification

| Attribute | Evidence |
|---|---|
| Correctness | 328 automated tests across parsing, quality, traceability, SysML comparison, PlusCal, LLM guards, agents, MCP, CLI and GUI endpoints |
| Round-trip | Tests assert regenerated text equals input |
| LLM safety | Tests with a scripted model: candidate status, severity cap, dropped unknown ids, failure recorded, call cap |
| Agent isolation | Test where one agent raises and the rest still report |
| Integration | Real LangGraph run, MCP protocol session, CLI run with exit code, GUI endpoint, browser check of card 7 |

---

## 8. Risks and known limitations

| Risk | Effect | Mitigation / next step |
|---|---|---|
| Heuristic checks can be wrong (word-overlap flow-down, name-based allocation) | False positives or misses | Marked `candidate`; make allocations explicit; use LLM or engineer review |
| SysML v2 pattern reader misses syntax | Missing structure | Extend patterns; consider a real KerML parser |
| Rule extraction misparses unusual sentences | Odd graph, wrong subject | Choose spaCy / BERT backend; review the tiles |
| LLM quality varies by model | Noisy candidates | Findings stay candidate-only; cap on calls |
| No persistence or multi-user | No baselines or history | Production path in the roadmap page: service deployment, storage, access control |
| Verification-method suggestion is a heuristic | Wrong default method | Engineer overrides in the RVTM |

---

## 9. Evolution

Listed in order of expected value; detail and workarounds are on the constraints and roadmap page
in the GUI.

1. Persist review runs and baselines, and compare runs over time.
2. Explicit human accept / reject of candidate findings, written back to the requirement source.
3. A full SysML v2 parser and API (not only text patterns).
4. Server deployment with authentication, for team use.
5. Domain rule packs (for example aviation ARP4754B / DO-178C) as configurable agent settings.
6. A review agent for verification coverage (requirements against test cases).
