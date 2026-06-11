# reqgraph — Complete Manual

IREB-CPRE aligned **Requirement ⇄ Graph** toolkit: turn any textual requirement
into a typed semantic graph and regenerate the *exact* text back, with pluggable
rule / spaCy / BERT extraction, quality analysis, duplicate-conflict screening,
and CSV/Excel/ReqIF I/O.

> Quick reference lives in [README.md](../README.md). This manual is the full
> reference: concepts → user guide → CLI → **complete API for every function** →
> data formats → limitations → extending.

---

## Table of contents
1. [Concepts](#1-concepts)
2. [Installation](#2-installation)
3. [User guide](#3-user-guide)
4. [CLI reference](#4-cli-reference)
5. [API reference (all functions)](#5-api-reference-all-functions)
6. [Data formats](#6-data-formats)
7. [Limitations](#7-limitations)
8. [Extending the toolkit](#8-extending-the-toolkit)

---

## 1. Concepts

### 1.1 Lossless tiling (the central guarantee)
The source sentence is **tiled**: every character is owned by exactly one
*terminal* node — either a semantic element (`CONDITION`, `SUBJECT`, …) or a
`GLUE` node holding separators. `graph.generate()` concatenates the terminals in
order, so the output equals the input **byte-for-byte**, no matter how the
elements were extracted. Extraction can be neural and imperfect; the text can
never be corrupted.

```
text ──(Extractor: rules | spaCy | BERT)──▶ claims (typed char spans)
     ──(lossless Tiler)──▶ RequirementGraph ──(generate)──▶ identical text
```

### 1.2 Element roles (IREB / Rupp MASTeR)
| Role | Meaning |
|---|---|
| `CONDITION` | logical/temporal pre- or post-condition (When/If/While…) |
| `SUBJECT` | the system-under-consideration / responsible actor |
| `MODALITY` | shall/should/will/may/must → legal obligation |
| `ACTOR` | the "whom" of a user-interaction (Rupp type-2) requirement |
| `PROCESS` | the process verb |
| `OBJECT` | the object the process acts on |
| `DETAILS` | additional object details |
| `CONSTRAINT` | quality/performance constraint (within 2 s, at 50 Hz…) |
| `ACTION` | group node bundling one PROCESS + its OBJECT(s) |
| `OPERATOR` | logical AND/OR joining compound actions |
| `ROOT` | the requirement as a whole |
| `GLUE` | literal separators (round-trip only; not semantic) |

### 1.3 Relationships (edges)
`HAS_CONDITION`, `HAS_SUBJECT`, `HAS_MODALITY`, `HAS_ACTION`, `HAS_ACTOR`,
`HAS_PROCESS`, `ACTS_ON`, `HAS_DETAILS`, `HAS_CONSTRAINT`, `OPERAND` (operator→
operand), `NEXT` (ordered terminal chain in text order).

### 1.4 Architecture
```
core.py        graph model + exporters
templates.py   Template + RUPP + EARS + register_template
tiling.py      tile_to_graph()  — lossless engine
parser.py      RequirementParser + build_requirement
extractors.py  Extractor ABC + Rule / spaCy / BERT + registry
nlp.py         BertTokenTagger (trainable) + RequirementAnalyzer
quality.py     IREB smells + type + EARS classification
seed_data.py   labelled aerospace corpus for the tagger
io_formats.py  CSV / Excel / ReqIF
__main__.py    command-line interface
```

---

## 2. Installation

| Capability | Install |
|---|---|
| Core (rules, graph, exporters) | pure Python + `numpy`, `networkx` |
| spaCy backend | `pip install spacy` + `python -m spacy download en_core_web_sm` |
| BERT tagger / analyzer | `pip install torch transformers` |
| CSV / Excel | `pip install pandas openpyxl` |
| ReqIF | `pip install lxml` |
| Tests | `pip install pytest` |

All optional imports are lazy: the package works for rule-based parsing even if
torch/spaCy/pandas are absent. Put the `reqgraph/` folder on your `PYTHONPATH`
(e.g. run from its parent directory).

---

## 3. User guide

### 3.1 Parse a requirement
```python
from reqgraph import RequirementParser, RUPP_TEMPLATE

text = ("When the cabin altitude exceeds 14,000 feet, the oxygen system shall "
        "deploy the passenger oxygen masks within 4 seconds.")
g = RequirementParser(RUPP_TEMPLATE).parse(text, metadata={"id": "REQ-002"})

print(g.summary())
assert g.generate() == text          # lossless, always
for n in g.elements():
    print(n.role.value, ":", n.text.strip())
```

### 3.2 Choose a backend
```python
from reqgraph.extractors import RuleExtractor, SpacyExtractor, BertTaggerExtractor

RequirementParser(RUPP_TEMPLATE, RuleExtractor())                 # default, deterministic
RequirementParser(RUPP_TEMPLATE, SpacyExtractor())                # dependency parse
RequirementParser(RUPP_TEMPLATE, BertTaggerExtractor(model_dir="models/req_tagger"))
```
Pick `RuleExtractor` when you need repeatable, certifiable output; `SpacyExtractor`
for free-form prose; `BertTaggerExtractor` once you have a trained model.

### 3.3 Read the graph
```python
g.elements()                 # semantic terminals in text order
g.by_role(Role.CONSTRAINT)   # only constraints
g.out_edges(g.root_id)       # edges leaving the ROOT
g.nodes, g.edges             # raw maps/lists
```

### 3.4 Export / visualise
```python
g.to_json()        # canonical, reload with RequirementGraph.from_dict
g.to_mermaid()     # paste into mermaid.live or Markdown
g.to_dot()         # Graphviz:  dot -Tpng
g.to_cypher()      # Neo4j ingestion
g.to_networkx()    # networkx.DiGraph (needs networkx)
```

### 3.5 Custom requirement structure
```python
from reqgraph import Template, register_template, Role, RequirementParser

contract = Template(
    name="CONTRACT",
    modality_keywords=("is required to", "is permitted to", "is prohibited from"),
    user_interaction_open=(), interface_markers=(),
    slot_order=(Role.CONDITION, Role.SUBJECT, Role.MODALITY,
                Role.PROCESS, Role.OBJECT, Role.CONSTRAINT),
)
register_template(contract)
RequirementParser(contract).parse(
    "The ground station is required to acknowledge the packet within 2 seconds.")
```

### 3.6 Author a new requirement from elements
```python
from reqgraph import build_requirement, Role
g = build_requirement({
    Role.CONDITION:  "When the landing gear is selected down",
    Role.SUBJECT:    "the landing gear control system",
    Role.MODALITY:   "shall",
    Role.PROCESS:    "extend",
    Role.OBJECT:     "the main landing gear",
    Role.CONSTRAINT: "within 10 seconds",
})
print(g.generate())
```

### 3.7 Quality, type & EARS analysis
```python
from reqgraph.quality import enrich
g = enrich(RequirementParser(RUPP_TEMPLATE).parse(text))
g.analysis["type"]          # functional/performance/interface/safety/usability
g.analysis["ears_pattern"]  # ubiquitous/event/state/unwanted/optional
g.analysis["quality"]       # weak_words, passive_voice, missing_modality, ...
```

### 3.8 Duplicate / conflict screening (BERT embeddings)
```python
from reqgraph.nlp import RequirementAnalyzer
az = RequirementAnalyzer("prajjwal1/bert-tiny")   # or "bert-base-uncased"
az.find_duplicates(catalogue, threshold=0.95)     # [(i, j, score), ...]
az.detect_conflicts(catalogue, threshold=0.9)     # similar + opposite polarity
az.similarity(a, b)                               # cosine in [0, 1]
```

### 3.9 Train the BERT element tagger
```python
from reqgraph.nlp import BertTokenTagger
from reqgraph.seed_data import training_examples

# ship-in aerospace corpus (or supply your own [(text, [(s, e, Role)...]), ...])
tagger = BertTokenTagger("bert-base-uncased").train(training_examples(), epochs=80)
tagger.save("models/req_tagger")

# reuse
tagger = BertTokenTagger.load("models/req_tagger")
tagger.predict_spans("The system shall measure the cabin pressure.")
```
`bert-tiny` (default) trains in seconds and is fine for a smoke test; switch to
`bert-base-uncased` for production accuracy.

### 3.10 Batch CSV / Excel / ReqIF
```python
from reqgraph.io_formats import (write_csv, write_excel, write_reqif,
                                 read_requirements_csv, read_reqif,
                                 requirements_to_dataframe)
items = [("REQ-1", "The autopilot shall maintain the heading."), ...]
write_csv(items,  "out.csv")     # decomposed element columns + type/EARS/smells
write_excel(items,"out.xlsx")
write_reqif(items,"out.reqif")
df = requirements_to_dataframe(items)            # pandas DataFrame
items2 = read_requirements_csv("out.csv")        # [(id, text), ...]
```

---

## 4. CLI reference

```
python -m reqgraph <command> [options]
```

### gui — local web GUI
```
python -m reqgraph gui [--port 8765] [--model <bert-dir>] [--no-browser]
```
Starts a zero-dependency local web app (stdlib `http.server`, **127.0.0.1
only**, works offline) and opens your browser. Features: requirement input with
examples, template + backend switching (rules / spaCy / BERT — availability
auto-detected), KPI cards (round-trip, quality score 0–100, element count,
text coverage %, IREB type, EARS pattern, obligation, parse time, atomicity),
the color-tiled requirement text, the semantic graph as an interactive-light
SVG tree, an element table, and SVG/JSON/Mermaid/DOT downloads.
API (for scripting): `GET /api/info`, `POST /api/parse {text, template?, backend?}`.
Python: `from reqgraph.gui import launch, make_server, parse_request`.

### parse — one requirement
```
python -m reqgraph parse "<text>"
        [--template IREB-Rupp|EARS|<name>]
        [--backend rules|spacy|bert] [--model <dir>]
        [--format summary|json|mermaid|dot|cypher|elements]
        [--analyze] [--id <REQ-ID>]
```
Example:
```
python -m reqgraph parse "When a fault is detected, the controller shall isolate the unit within 100 ms." --analyze
python -m reqgraph parse "The system shall close the valve." --backend bert --model models/req_tagger --format elements
```

### batch — a CSV/Excel/ReqIF set
```
python -m reqgraph batch <in.csv|in.xlsx|in.reqif>
        [--out out.csv|.xlsx|.reqif|.json]
        [--template ...] [--backend ...] [--model ...]
```
No `--out` prints a table. Output format is inferred from the extension.

### train — fine-tune & save a tagger
```
python -m reqgraph train [--model prajjwal1/bert-tiny|bert-base-uncased]
        [--epochs 60] [--lr 5e-4] [--out models/req_tagger]
        [--data labelled.jsonl]
```
Uses the built-in seed corpus unless `--data` (JSONL, see §6.3) is given.

### analyze — quality + duplicates/conflicts for a set
```
python -m reqgraph analyze <in.csv|in.xlsx|in.reqif>
        [--model <embedding model>]
        [--dup-threshold 0.95] [--conflict-threshold 0.9]
```

---

## 5. API reference (all functions)

### 5.1 `reqgraph.core`

**`Role(str, Enum)`** — element roles: `ROOT, CONDITION, SUBJECT, MODALITY,
ACTION, ACTOR, PROCESS, OBJECT, DETAILS, CONSTRAINT, OPERATOR, GLUE`.

**`Rel(str, Enum)`** — edge types (see §1.3).

**`OBLIGATION: dict`** — maps a modal verb (`"shall"`, `"should"`, …) to its IREB
legal-obligation description.

**`Node`** — `Node(id, role, text="", attrs={})`. Property `is_terminal` →
`True` for text-bearing roles.

**`Edge`** — `Edge(source, target, rel, attrs={})` (source/target are node ids).

**`RequirementGraph(template_name="", metadata=None)`** — the graph container.
Attributes: `nodes` (id→Node), `edges` (list), `leaf_order` (terminal ids in text
order), `root_id`, `template_name`, `metadata`, `analysis`.

| Method | Description |
|---|---|
| `new_id(role)` | generate a unique node id |
| `add_node(role, text="", **attrs)` | create + register a node, returns it |
| `add_edge(source, target, rel, **attrs)` | create + register an edge |
| `generate() -> str` | **regenerate the requirement text (lossless)** |
| `elements() -> list[Node]` | semantic (non-glue) terminals in text order |
| `by_role(role) -> list[Node]` | semantic terminals of one role |
| `out_edges(node_id) -> list[Edge]` | edges leaving a node |
| `to_dict()` / `to_json(indent=2)` | serialise |
| `from_dict(d)` *(classmethod)* | deserialise |
| `to_mermaid(show_glue=False)` | Mermaid flowchart string |
| `to_dot(show_glue=False)` | Graphviz DOT string |
| `to_cypher()` | Neo4j CREATE/MATCH statements |
| `to_networkx()` | networkx.DiGraph (needs networkx) |
| `summary() -> str` | human-readable element table |

### 5.2 `reqgraph.templates`

**`Template(name, …)`** — dataclass configuring a requirement structure. Fields:
`condition_markers`, `modality_keywords`, `user_interaction_open`,
`user_interaction_bridge`, `interface_markers`, `constraint_markers`,
`object_determiners`, `conjunctions`, `slot_order`. Method `alt(items)` builds a
longest-first regex alternation.

**`RUPP_TEMPLATE`**, **`EARS_TEMPLATE`** — ready-made templates.
**`TEMPLATES: dict`** — name → Template registry.
**`register_template(template)`** — add a custom template to the registry.

### 5.3 `reqgraph.tiling`

**`Claim`** — `(start: int, end: int, Role, attrs: dict)`.

**`tile_to_graph(text, claims, template_name="", metadata=None) -> RequirementGraph`**
— the lossless engine: dedups/sorts claims, fills gaps with `GLUE`, builds the
terminal backbone and the semantic tree (ROOT → ACTION → PROCESS/OBJECT, AND/OR
operators for compound actions, conditions, constraints). Operator type is
inferred from the connective text between actions.

### 5.4 `reqgraph.parser`

**`RequirementParser(template=RUPP_TEMPLATE, extractor=None)`** — orchestrator.
- `parse(text, metadata=None) -> RequirementGraph`
- `roundtrip_ok(text) -> bool`

**`build_requirement(elements, template=RUPP_TEMPLATE, metadata=None, extractor=None)`**
— synthesise a requirement graph from an `{Role: text}` (or `{"NAME": text}`)
mapping, laid out in the template's `slot_order`.

### 5.5 `reqgraph.extractors`

**`Extractor`** *(ABC)* — `available() -> bool`, `extract(text, template) ->
list[Claim]`.

**`RuleExtractor()`** — deterministic anchor heuristics (default). Always
available.

**`SpacyExtractor(model="en_core_web_sm")`** — dependency-parse extraction.
`available()` checks the spaCy model is installed.

**`BertTaggerExtractor(tagger=None, model_dir=None)`** — wraps a
`BertTokenTagger`; pass a live `tagger` or a saved `model_dir`.

**`get_extractor(name, **kwargs) -> Extractor`** — factory (`"rules"|"spacy"|"bert"`).
**`auto_select() -> Extractor`** — best no-training backend (spaCy else rules).
**`EXTRACTORS: dict`** — name → class registry.

### 5.6 `reqgraph.nlp`

**`TAG_ROLES`, `LABELS`, `LABEL2ID`, `ID2LABEL`** — the BIO label scheme.

**`BertTokenTagger(model_name="prajjwal1/bert-tiny", tokenizer_name=None,
device=None, max_len=64)`** — fine-tunable token classifier.
- `train(examples, epochs=15, lr=5e-4, batch_size=8, verbose=True)` — `examples`
  = `[(text, [(start, end, Role), ...]), ...]`. Returns self.
- `predict_spans(text) -> list[(start, end, Role)]`
- `save(path) -> path`
- `load(path, device=None)` *(classmethod)*

**`RequirementAnalyzer(model_name="prajjwal1/bert-tiny", tokenizer_name=None,
device=None, max_len=64)`** — embedding intelligence.
- `embed(texts) -> np.ndarray` (mean-pooled, L2-normalised)
- `similarity(a, b) -> float`
- `find_duplicates(reqs, threshold=0.9) -> [(i, j, score), ...]`
- `detect_conflicts(reqs, threshold=0.8) -> [(i, j, score), ...]`

### 5.7 `reqgraph.quality`

- **`check_quality(text) -> dict`** — keys: `weak_words` (list), `passive_voice`,
  `missing_modality`, `vague_quantifier`, `non_atomic`.
- **`classify_type(graph_or_text) -> str`** — functional/performance/interface/
  safety/usability.
- **`classify_ears(graph) -> str`** — ubiquitous/event/state/unwanted/optional.
- **`enrich(graph) -> graph`** — writes the three above into `graph.analysis`.
- **`WEAK_WORDS`, `TYPE_KEYWORDS`** — the underlying word lists (editable).

### 5.8 `reqgraph.seed_data`

- **`training_examples() -> list`** — the labelled aerospace corpus as
  `[(text, [(start, end, Role), ...]), ...]`.
- **`validate() -> int`** — assert all labels are unambiguous; return the count.

### 5.9 `reqgraph.io_formats`

- **`requirements_to_dataframe(items, template=RUPP_TEMPLATE, extractor=None,
  analyze=True)`** — pandas DataFrame with one column per element + type/EARS/
  smells. `items` = `[text, ...]` or `[(id, text), ...]`.
- **`write_csv(items, path, **kw)`**, **`write_excel(items, path, **kw)`**
- **`read_requirements_csv(path, text_column="text", id_column="id")`**,
  **`read_requirements_excel(...)`** — return `[(id, text), ...]`.
- **`write_reqif(items, path, title="reqgraph export")`** — minimal ReqIF XML.
- **`read_reqif(path)`** — return `[(id, text), ...]`.

---

## 6. Data formats

### 6.1 Graph JSON
`to_dict()`/`to_json()` produce: `template`, `metadata`, `analysis`, `root`,
`leaf_order` (ordered terminal ids — the lossless backbone), `nodes`
(`id, role, text, attrs`), `edges` (`source, target, rel, attrs`). Reload with
`RequirementGraph.from_dict`.

### 6.2 Claim tuple
Extractors emit `[(start, end, Role, attrs), ...]`. Overlaps are resolved
earliest-first; gaps become `GLUE`. `function_type` on a `PROCESS` claim flows
onto its `ACTION` node.

### 6.3 Training JSONL (`reqgraph train --data`)
One JSON object per line:
```json
{"text": "The sensor shall measure the cabin pressure.",
 "spans": [[0,10,"SUBJECT"],[11,16,"MODALITY"],[17,24,"PROCESS"],[25,43,"OBJECT"]]}
```

### 6.4 ReqIF subset
A well-formed REQ-IF document with one string datatype, a `Requirement`
spec-object-type carrying `ReqID` + `ReqText` attributes, and one `SPEC-OBJECT`
per requirement. Export→import round-trips the text exactly. Extend the attribute
set for richer DOORS/Polarion interchange.

---

## 7. Limitations

1. **Lossless round-trip is unconditional; semantic labelling is best-effort.**
   A mislabelled element never changes the regenerated text — only its role tag.
2. **Rule backend vs prose.** The rule extractor is most precise on the Rupp/EARS
   boilerplate. On free, multi-clause prose its `PROCESS`/`OBJECT` split or nested
   conditions may be wrong; prefer spaCy or a trained BERT tagger there.
3. **spaCy on Rupp type-2.** "provide ‹whom› with the ability to ‹verb›" makes the
   root verb `provide`, so spaCy reports `provide` as the process and may miss the
   real verb after "to". The rule backend handles this form better.
4. **bert-tiny is a smoke model.** Its embeddings are weak (it missed an
   autopilot↔auto-flight paraphrase at cosine 0.95) and a tagger trained on it
   only generalises near the seed distribution. Use `bert-base-uncased` (or a
   domain model) + more labelled data for production.
5. **Seed corpus is small (30 reqs).** Great for the boilerplate styles it covers;
   add your own labelled requirements for unusual phrasings, units, or domains.
6. **Single-sentence assumption.** One requirement = one sentence. Multi-sentence
   or tabular requirements should be split first.
7. **Heuristic classifiers.** `classify_type`/`classify_ears` and the quality
   smells are keyword/marker based — fast and explainable, but tune
   `TYPE_KEYWORDS`/`WEAK_WORDS` for your house style.
8. **Conflict detection is a screen, not a proof.** "similar + opposite polarity"
   surfaces candidates for human review; it does not formally verify consistency.
9. **ReqIF is a minimal subset.** IDs + text only; not a full datatype/spec-type
   interchange.
10. **English only** (markers, spaCy model, BERT vocab).

---

## 8. Extending the toolkit

### 8.1 A custom extraction backend
```python
from reqgraph.extractors import Extractor
from reqgraph import Role, RequirementParser, RUPP_TEMPLATE

class MyExtractor(Extractor):
    name = "mine"
    def extract(self, text, template):
        # return [(start, end, Role, attrs), ...]; tiler guarantees losslessness
        return [(0, len(text.split()[0]), Role.SUBJECT, {})]

RequirementParser(RUPP_TEMPLATE, MyExtractor()).parse("...")
```

### 8.2 A custom template
Subclass-free — just construct `Template(...)` with your marker sets and
`slot_order`, then `register_template`. See §3.5.

### 8.3 Custom roles
Add a value to `Role` in `core.py`, include it in `TERMINAL_ROLES` if it carries
text, teach an extractor to emit it, and (optionally) wire an edge in
`tiling.py`. The lossless guarantee is unaffected.
```
