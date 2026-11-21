---
name: raven-requirements-review
description: Review a requirement set, and optionally its SysML architecture, with RAVEN's review agents. Checks writing quality, completeness, rationale, source and verification method, consistency, the requirement cascade (parent/child decomposition, budgets) and, given a model, requirement-to-architecture allocation, unjustified elements and missing derived requirements. Use when asked to review, baseline, gate-check or audit requirements (ReqIF, CSV, Excel, JSON, Word), to check requirements against a SysML v2 or v1 model, or to find missing derived requirements.
---

# RAVEN requirements review

RAVEN (`reqgraph` package) runs eight review agents:

- **Quality team** (requirement set only): `writing_quality`, `completeness`, `rationale`,
  `consistency`.
- **Cascade:** `cascade`. It needs parent or derived-from links or a level column. ReqIF
  relations are read automatically.
- **Architecture team** (needs a SysML model): `allocation`, `architecture_justification`,
  `derived_requirements`.

## Steps

1. **Find the inputs.**
   - The requirement file: `.reqif`, `.csv`, `.xlsx`, `.json`, `.docx` or `.txt`.
   - The architecture model, if there is one: `.sysml` (v2), or `.xmi` / `.ttl` / `.mdzip` (v1).

   Columns such as `rationale`, `parent`, `level`, `verification_method` and `allocated_to` are
   used when present.
2. **Run the review.** Use one of these:
   - **MCP server**, if `raven` is configured: call `raven_review` with `requirements_path`
     and, optionally, `architecture_path`. `raven_quality_review`, `raven_cascade_review` and
     `raven_architecture_review` each run one team.
   - **CLI:**
     ```bash
     python -m reqgraph review <reqs> [-a <model>] --format json -o review.json
     ```
     Add `--llm --provider ollama|vertex` to get AI candidate findings.
   - **Python:**
     ```python
     from reqgraph.agents import review
     from reqgraph.architecture import load_architecture
     report = review(items, load_architecture(path))
     ```
3. **Read the report.**
   - `summary.verdict` and the ready / needs work / blocked counts.
   - `summary.agents_skipped`, which says why an agent did not run (for example, "no
     architecture model supplied").
   - `findings`, sorted by severity.
   - `elements`, the architecture-side issues.
4. **Report to the user.**
   - Give the verdict.
   - Give the main issues grouped by team: quality, cascade, architecture.
   - Name the requirements or elements most in need of attention.
   - Give concrete next actions, using each finding's `hint`.
   - Label `status: "candidate"` findings as needing engineer confirmation. These are
     heuristic matches and LLM suggestions.

## Interpreting key rules

- `missing_rationale`, `trivial_rationale`, `restating_rationale`: every requirement needs a
  justification that says *why*: the need, regulation, hazard or analysis behind it.
- `orphan`, `dangling_parent`, `not_decomposed`, `weak_flowdown`, `budget_violation`: the
  cascade is broken or insufficient. A child bound must be at least as tight as its parent's.
- `unallocated`: no element realises the requirement. Either the architecture is missing
  something or the requirement is out of scope.
- `unjustified_element`: the model has a part that no requirement drives. This is either
  gold-plating or a missing requirement.
- `model_requirement_missing`: a requirement is written in the model but not in the
  specification.
- `derived_requirement_expected`: the decomposition created a component with no derived
  requirement.
- `allocation_outside_parent_branch`: a child requirement is allocated outside its parent's
  architecture branch. It may be traced to the wrong parent.

## Do not

- Do not present candidate findings as confirmed.
- Do not invent requirement ids or element names that are not in the report.
- Do not edit the user's requirement files unless asked. Propose rewrites instead.
