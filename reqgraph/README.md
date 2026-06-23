# reqgraph — IREB-CPRE Requirement ⇄ Graph toolkit

Convert any textual requirement into a typed semantic **graph** (nodes + edges)
and regenerate the **exact** original text from the graph. Modular extraction
backends (rules / spaCy / BERT), IREB/Rupp + EARS + custom templates, quality
analysis, duplicate/conflict detection, and CSV/Excel/ReqIF batch I/O.

📖 **Full reference (every function, user guide, limitations):**
[`docs/MANUAL.md`](docs/MANUAL.md)
🏗️ **Architecture & function-level data flow:**
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

Install (editable, with console script):
```bash
pip install -e .            # core only — zero dependencies
pip install -e .[all]       # spaCy + torch/transformers + pandas/lxml + networkx
reqgraph --version          # or: python -m reqgraph --version
```

## The core guarantee: lossless round-trip

The source sentence is **tiled** — every character is owned by exactly one
terminal node (a semantic element such as `CONDITION`/`SUBJECT`/`PROCESS`, or a
`GLUE` node for separators). So `graph.generate()` reproduces the input
**byte-for-byte**, regardless of how the elements were extracted. Extraction can
be as "fuzzy" as a neural net; the text can never be corrupted.

```
text ──(Extractor: rules | spaCy | BERT)──▶ claims (typed char spans)
     ──(lossless Tiler)──▶ RequirementGraph ──(generate)──▶ identical text
```

## Install

Core (always): pure-Python, plus `numpy`/`networkx` for exports.

| Feature | Packages |
|---|---|
| spaCy backend | `pip install spacy` then `python -m spacy download en_core_web_sm` |
| BERT tagger + analyzer | `pip install torch transformers` |
| CSV / Excel | `pip install pandas openpyxl` |
| ReqIF | `pip install lxml` |
| tests | `pip install pytest` |

## Quick start

```python
from reqgraph import RequirementParser, RUPP_TEMPLATE

text = ("When the cabin altitude exceeds 14,000 feet, the oxygen system shall "
        "deploy the passenger oxygen masks within 4 seconds.")

g = RequirementParser(RUPP_TEMPLATE).parse(text, metadata={"id": "REQ-002"})
print(g.summary())
assert g.generate() == text          # lossless, always

print(g.to_mermaid())                # also: to_dot(), to_cypher(), to_networkx(), to_json()
print(g.to_graphml())                # GraphML knowledge graph (yEd / Gephi / networkx)
print(g.to_turtle())                 # RDF Turtle (.ttl) for triple stores / SPARQL
```

### Swap the extraction backend

```python
from reqgraph.extractors import SpacyExtractor
RequirementParser(RUPP_TEMPLATE, SpacyExtractor()).parse(text)
```

The default is the deterministic **rule** backend (repeatable — preferred where
traceability matters). spaCy adds dependency-parse intelligence for complex /
nested sentences.

### Fine-tune a BERT element tagger

```python
from reqgraph.nlp import BertTokenTagger
from reqgraph.extractors import BertTaggerExtractor
from reqgraph import Role, RequirementParser, RUPP_TEMPLATE

# label your own requirements as (text, [(start, end, Role), ...])
tagger = BertTokenTagger("prajjwal1/bert-tiny").train(my_labelled_data, epochs=40)
tagger.save("models/req_tagger")

parser = RequirementParser(RUPP_TEMPLATE, BertTaggerExtractor(tagger=tagger))
g = parser.parse("When the pressure drops, the controller shall close the valve.")
```

For production accuracy use `bert-base-uncased` (or a domain model) and a few
hundred labelled requirements. `bert-tiny` is the fast smoke-test default.

---

## Extraction backends

All three backends implement the same `Extractor` interface and return a list of
**Claims** — typed character spans `(start, end, Role, attrs)`. The lossless
tiler converts claims into the graph, so swapping backends never changes the
round-trip guarantee; at worst, a weaker backend mislabels some spans.

```
text ──(Extractor)──▶ Claims ──(lossless Tiler)──▶ RequirementGraph
                     ^^^^^^^^
          every backend must only propose spans;
          corruption is architecturally impossible
```

### Rules backend (default)

**Import:** `from reqgraph.extractors import RuleExtractor`  
**Dependencies:** none — pure Python stdlib  
**Speed:** ~0.05 ms per requirement (~22,000 req/s)

The rule backend uses the keyword sets from your `Template` (condition markers,
modality keywords, constraint markers, etc.) compiled into regular expressions
cached once per template. It works by progressively **narrowing a search
window** through six stages:

| Stage | What it finds | How |
|---|---|---|
| 1 | Leading **condition** | `cond_lead` regex at text start; ends at the last comma before the modality (protects "14,000 feet") or at `then` |
| 2 | **Modality** + **Subject** | First modality keyword after the condition; everything between the condition and the modality becomes the Subject |
| 3 | Trailing **condition** | Condition marker after the modality (e.g. `"if the parking brake is set"`) |
| 4 | Trailing **constraint** | Constraint marker after the modality (e.g. `"within 4 seconds"`) |
| 5 | **Function type** | Rupp type-2 pattern `"provide <whom> with the ability to"` → `ACTOR`; `"be able to"` → interface requirement |
| 6 | **Action core** | Top-level conjunction followed by a non-determiner word splits two actions; first determiner within an action splits `PROCESS` from `OBJECT` |

**Best for:** boilerplate IREB/Rupp and EARS requirements; CI pipelines where
determinism and zero dependencies matter; audit trails that require repeatable
output.

**Limitations:** rigid keyword anchoring — unusual word order or heavily
subordinated clauses can fool the window logic. Free-prose requirements with
no recognised modality yield mostly `GLUE` (text is still preserved losslessly).

```python
from reqgraph import RequirementParser, RUPP_TEMPLATE
from reqgraph.extractors import RuleExtractor

g = RequirementParser(RUPP_TEMPLATE, RuleExtractor()).parse(
    "When the cabin altitude exceeds 14,000 feet, "
    "the oxygen system shall deploy the passenger oxygen masks within 4 seconds.")
# → CONDITION / SUBJECT / MODALITY / PROCESS / OBJECT / CONSTRAINT — all correct
```

---

### spaCy backend

**Import:** `from reqgraph.extractors import SpacyExtractor`  
**Dependencies:** `pip install spacy && python -m spacy download en_core_web_sm`  
**Speed:** ~5–15 ms per requirement (spaCy pipeline loaded once, cached)

The spaCy backend runs the full spaCy NLP pipeline to obtain a **dependency
parse tree**, then maps linguistic relations to IREB roles:

| Dependency relation | Maps to |
|---|---|
| `advcl` subtree (subordinate adverbial clause) | `CONDITION` — pre- or post-condition depending on position relative to the ROOT verb |
| `prep` headed by a constraint marker | `CONSTRAINT` |
| Rupp "provide X with the ability to" pattern | `ACTOR` (the `X` span) |
| ROOT verb's `aux`/`auxpass` child with modal tag `MD` | `MODALITY` |
| ROOT verb's `nsubj`/`nsubjpass` subtree | `SUBJECT` |
| ROOT verb + all `conj` verbs (coordinated root verbs) | one `PROCESS` each |
| `dobj`/`obj`/`attr` children of each verb | `OBJECT` |
| `dative` children | `ACTOR` |

If spaCy finds no ROOT token the backend falls back to `RuleExtractor`
automatically.

**Best for:** complex sentences with deeply embedded clauses, passive voice,
co-ordinated verbs, or free prose that does not follow IREB boilerplate.

**Limitations:** weaker on the Rupp "provide … with the ability to" construction
(mitigated by the regex fallback inside the backend); heavier runtime (~80 MB
model in memory); non-deterministic across spaCy versions.

```python
from reqgraph import RequirementParser, RUPP_TEMPLATE
from reqgraph.extractors import SpacyExtractor

g = RequirementParser(RUPP_TEMPLATE, SpacyExtractor()).parse(
    "While the aircraft is on the ground, the avionics suite shall provide "
    "the pilot with the ability to configure the flight plan "
    "if the parking brake is set.")
# dependency tree correctly identifies the nested conditions and actor
```

To use a larger, more accurate model:

```python
SpacyExtractor(model="en_core_web_trf")   # transformer-based, much more accurate
```

---

### BERT token-tagger backend

**Import:** `from reqgraph.extractors import BertTaggerExtractor`  
**Dependencies:** `pip install torch transformers`  
**Speed:** ~10–30 ms per requirement on CPU (model loaded once; GPU used automatically)

The BERT backend fine-tunes a **token classifier** on your labelled requirements
and uses it to assign an IREB role to every WordPiece token. It works in two
phases:

#### Training phase

```
labelled data  →  BertTokenTagger.train()
  [(text, [(start, end, Role), ...]), ...]
       │
       ▼
  tokenizer(text, return_offsets_mapping=True)
       │  maps character spans → BIO token labels
       │  (B-SUBJECT, I-SUBJECT, B-MODALITY, ..., O)
       │  special tokens ([CLS], [SEP]) → label -100 (ignored by loss)
       ▼
  AutoModelForTokenClassification (BERT + linear head)
  trained with AdamW + CrossEntropy over BIO labels
       │
       ▼
  model.eval() → save_pretrained(dir)
```

#### Inference phase

```
text  →  tokenizer(offsets)  →  model logits [1, T, |labels|]
       →  argmax per token  →  BIO label sequence
       →  span reconstruction via offset mapping
       →  [(start, end, Role)]  →  Claims for the tiler
```

**Choosing a base model:**

| Checkpoint | Use case |
|---|---|
| `prajjwal1/bert-tiny` | Smoke-test / CI (ships as default; weak accuracy) |
| `bert-base-uncased` | General production use (recommended starting point) |
| `bert-base-cased` | Domain text where case carries meaning (e.g. product names) |
| Domain-specific BERT | Best accuracy when pre-trained on requirements engineering or your industry corpus |

**Training data:** the built-in seed corpus (`reqgraph/seed_data.py`) contains
30 labelled aerospace requirements. For production-grade accuracy, annotate a
few hundred requirements from your own domain using the same `(text, spans)`
format and pass them to `BertTokenTagger.train()`.

```python
from reqgraph.nlp import BertTokenTagger
from reqgraph.extractors import BertTaggerExtractor
from reqgraph import RequirementParser, RUPP_TEMPLATE, Role

# Annotate your requirements
my_data = [
    ("The sensor shall measure the cabin pressure.",
     [(0, 10, Role.SUBJECT), (11, 16, Role.MODALITY),
      (17, 24, Role.PROCESS), (25, 44, Role.OBJECT)]),
    # ... hundreds more
]

# Train and save
tagger = BertTokenTagger("bert-base-uncased").train(my_data, epochs=40)
tagger.save("models/my_tagger")

# Use in the parser
g = RequirementParser(
    RUPP_TEMPLATE,
    BertTaggerExtractor(model_dir="models/my_tagger")
).parse("The controller shall isolate the faulty channel.")
```

**Via CLI:**

```bash
# Train on the built-in seed corpus
python -m reqgraph train --model bert-base-uncased --epochs 80 --out models/req_tagger

# Parse using the saved tagger
python -m reqgraph parse "The system shall close the valve." \
    --backend bert --model models/req_tagger --format elements
```

**Best for:** domain-specific vocabularies, requirements that don't follow IREB
boilerplate, high-volume pipelines where you can invest in labelling, or when
the rule/spaCy backends consistently mislabel the same patterns.

**Limitations:** requires labelled training data; `bert-tiny` is for testing
only; runs significantly slower than rules on CPU; output varies slightly with
PyTorch version and hardware.

---

### Choosing a backend

| Criterion | Rules | spaCy | BERT |
|---|---|---|---|
| Zero dependencies | ✓ | — | — |
| Deterministic output | ✓ | — | — |
| Complex/nested sentences | limited | ✓ | ✓ |
| Domain-specific vocabulary | limited | limited | ✓ (with training) |
| Speed | fastest | medium | slowest |
| Training data required | none | none | yes |
| Good for CI/traceability | ✓ | — | — |

When in doubt, start with **Rules**. Add **spaCy** if you have complex,
free-prose sentences. Switch to **BERT** only when you have labelled data and
the other two backends consistently produce incorrect labels for your domain.

---

## Compound requirement detection

When a single text contains more than one modality clause ("The system shall X
**and the controller shall Y**"), reqgraph automatically detects and splits it
into independent requirements before parsing each one separately.

```python
from reqgraph import RequirementParser, RUPP_TEMPLATE

p = RequirementParser(RUPP_TEMPLATE)

# auto-split into independent statements
segments = p.split(
    "The pump shall start within 2 seconds and the controller shall log the event.")
# → ['The pump shall start within 2 seconds',
#    'the controller shall log the event.']

# parse each segment into its own graph
graphs = p.parse_many(
    "The pump shall start within 2 seconds and the controller shall log the event.",
    metadata={"id": "REQ-10"})
# → two RequirementGraph objects; metadata ids become REQ-10-1, REQ-10-2

# or as a standalone helper
from reqgraph import split_requirements
split_requirements(text, template)
```

Compound actions under a single modality (`"shall shut off the engine and
activate the suppression system"`) are correctly kept as one atomic requirement.

The **quality checker** also flags unsplit compound requirements with a
`compound_requirement` smell, causing a penalty on the quality score.

**CLI:** `reqgraph parse` auto-splits and prints each requirement separately.
Pass `--no-split` to parse as a single unit.

**GUI:** an amber warning banner appears when compound input is detected, with
clickable tabs to navigate each split requirement's graph independently.

---

## Requirement sets: cross-requirement connections

Beyond a single requirement's graph, reqgraph can analyze a whole **set** of
requirements together: every item is parsed into its own graph (compound items
are split first, exactly like `parse`), then the SUBJECT/OBJECT elements are
compared *across* requirements to surface ones that govern the same component
or act on the same data — useful for traceability review and for spotting
hidden coupling that a flat list of requirements hides.

```python
from reqgraph.corpus import build_requirement_set_graph

reqs = [
    "The flight management system shall calculate the optimal cruise altitude.",
    "The flight management system shall log every altitude change to the maintenance recorder.",
    "The pilot shall be able to override the calculated cruise altitude.",
]
rsg = build_requirement_set_graph(reqs)

for c in rsg.connections:
    print(c.a.req_id, c.role.value, repr(c.a.text), "~", c.score, "~", repr(c.b.text), c.b.req_id)
# REQ-1 SUBJECT 'The flight management system' ~ 1.0  ~ 'The flight management system' REQ-2
# REQ-1 OBJECT  'the optimal cruise altitude'  ~ 0.73 ~ 'the calculated cruise altitude' REQ-3

print(rsg.to_mermaid())   # also: to_dot(), to_graphml(), to_turtle(), to_cypher(), to_dict()
```

`items` accepts the same shapes as `io_formats`: `["text", ...]` or
`[(id, text), ...]`.

**Similarity** is pluggable via `similarity=`:

* `"lexical"` (default) — token-overlap (Jaccard) blended with a character
  sequence ratio, with determiners (`the`/`a`/`their`/...) stripped before
  comparing so two unrelated "the X system" spans don't match just because
  both start with "the ... system". Zero dependencies.
* `"embedding"` — BERT cosine similarity via `reqgraph.nlp.RequirementAnalyzer`
  (requires `torch`+`transformers`); pass `embedding_model=` to choose the
  checkpoint.

Tune `threshold` (default `0.6`) to control how strict a match must be, and
`roles=(Role.SUBJECT, Role.OBJECT, ...)` to cross-reference other element
types (e.g. add `Role.ACTOR`).

**CLI:**

```bash
python -m reqgraph connections reqs.csv                       # human-readable report
python -m reqgraph connections reqs.csv --format json          # full payload
python -m reqgraph connections reqs.csv --format mermaid       # visualize
python -m reqgraph connections reqs.csv --similarity embedding --threshold 0.85
```

**GUI:** the "Requirement set — find connections" panel accepts one
requirement per line and renders the connections as a network graph (one node
per requirement, edges colored by SUBJECT/OBJECT and weighted by score), plus
the same Mermaid/DOT/GraphML/Turtle/Cypher export buttons as a single graph.

---

### Custom requirement structure (R4)

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

### Author a new requirement from elements

```python
from reqgraph import build_requirement, Role
g = build_requirement({
    Role.CONDITION: "When the landing gear is selected down",
    Role.SUBJECT:   "the landing gear control system",
    Role.MODALITY:  "shall",
    Role.PROCESS:   "extend",
    Role.OBJECT:    "the main landing gear",
    Role.CONSTRAINT:"within 10 seconds",
})
print(g.generate())
```

### Quality, type & EARS analysis (IREB)

```python
from reqgraph.quality import enrich
g = enrich(RequirementParser(RUPP_TEMPLATE).parse(text))
print(g.analysis["type"], g.analysis["ears_pattern"], g.analysis["quality"])
# flags weak words, passive voice, missing modality, vague quantifiers, non-atomic
```

### Duplicate / conflict screening (BERT embeddings)

```python
from reqgraph.nlp import RequirementAnalyzer
az = RequirementAnalyzer("prajjwal1/bert-tiny")
az.find_duplicates(catalogue, threshold=0.95)   # near-duplicates
az.detect_conflicts(catalogue, threshold=0.9)   # similar + opposite polarity
```

### Batch I/O

```python
from reqgraph.io_formats import write_csv, write_excel, write_reqif, read_reqif
write_csv(items, "reqs.csv")       # decomposed element columns + type/EARS/smells
write_excel(items, "reqs.xlsx")
write_reqif(items, "reqs.reqif")   # OMG ReqIF (DOORS/Polarion-friendly subset)
read_reqif("reqs.reqif")
```

All readers — `read_requirements_csv`, `read_requirements_excel`,
`read_requirements_json`, `read_reqif` — return `(id, text, metadata)` triples.
Any extra columns/attributes (e.g. **rationale**, **applicability**, **additional
info**) are captured into `metadata` with normalised names (lowercased,
spaces/hyphens → underscores). A source column whose name collides with a parser
output column (e.g. a `Subject` or `Type` attribute) is preserved under an
`attr_` prefix so no data is lost.

```python
from reqgraph.io_formats import read_requirements_json
items = read_requirements_json("reqs.json")   # [{"id","text","rationale",...}] or {"R1": {...}}
# items -> [("R1", "The system shall ...", {"rationale": "...", "applicability": "..."}), ...]
```

### One-shot export: CSV + JSON + consolidated GraphML

`reqgraph export` reads any supported format, runs the parser, quality analysis,
**and** cross-requirement connection detection, then writes a quality table
(CSV/JSON) plus a single **element-level GraphML** in one pass. Compound rows are
split into atomic requirements consistently across every output.

```bash
python -m reqgraph export reqs.csv  --out-prefix build/out      # → out.csv, out.json, out.graphml
python -m reqgraph export reqs.json --csv q.csv --graphml g.xml  # pick individual outputs
python -m reqgraph export reqs.reqif --out-prefix out/ --threshold 0.5
```

The consolidated GraphML (`RequirementSetGraph.to_element_graphml()`) is one graph
containing **REQ** nodes (with quality attributes + metadata), **ELEMENT** nodes
for every SUBJECT/OBJECT/CONDITION/… span (`HAS_ELEMENT` edges), and `SIMILAR_*`
cross-requirement edges weighted by similarity — so tools like Gephi / yEd /
Cytoscape show all connected subjects and objects across the whole set.

## GUI

```bash
python -m reqgraph gui          # opens http://127.0.0.1:8765 in your browser
```

A zero-dependency local web app (stdlib `http.server`, bound to 127.0.0.1, fully
offline): type or pick a requirement, switch template/backend, and see
**KPI cards** (round-trip, quality score, elements/coverage, type, EARS,
obligation, parse time), the **color-tiled requirement** (every character mapped
to its owning node), the **semantic graph** rendered as an SVG tree, and the
**element table**.

The page also includes:

* a **"how it works" panel** that explains the currently-selected backend
  (rules / spaCy / BERT) and template (IREB-Rupp / EARS) and updates live as you
  switch — so you can see what each configuration actually does;
* one-click export to **SVG, JSON, Mermaid, DOT** and the **knowledge-graph
  formats GraphML, Turtle (RDF), and Cypher**;
* an amber **compound-requirement warning** with per-requirement tabs when a
  single input contains several "shall" clauses;
* a **"Requirement set — find connections"** panel: paste a requirement set
  (one per line) and see a network graph connecting requirements that share
  similar SUBJECT/OBJECT elements, with the same knowledge-graph export
  buttons.
* an **"Import & analyze"** panel: upload a CSV / Excel / JSON / ReqIF file (or
  paste raw text), and get the per-requirement quality table (auto-discovering
  any extra metadata columns), the cross-requirement connections network, and
  one-click downloads of the CSV, JSON, and consolidated element-level GraphML.

## Knowledge graph export

Every graph can be exported as a standards-based knowledge graph for downstream
tooling — no extra dependencies required (both are built with the standard
library):

```python
g.to_graphml()   # GraphML (XML) — open in yEd, Gephi, Cytoscape, networkx
g.to_turtle()    # RDF Turtle (.ttl) — load into any triple store, query with SPARQL
g.to_cypher()    # Cypher CREATE statements — import into Neo4j
```

```bash
python -m reqgraph parse "The system shall close the valve within 5 seconds." \
        --format graphml > req.graphml
python -m reqgraph parse "The system shall close the valve within 5 seconds." \
        --format turtle  > req.ttl
```

In Turtle, each element becomes a typed RDF resource (`a rg:SUBJECT`, `a
rg:MODALITY`, …) carrying its text as `rdfs:label`/`rg:text` plus its attributes
(obligation, function type, condition kind), and relationships become predicates
(`rg:HAS_MODALITY`, `rg:ACTS_ON`, …). GraphML carries the same role/text/attrs on
nodes and the relationship type on edges. `GLUE` separator nodes are omitted by
default (pass `show_glue=True` to include them).

## Command line

```bash
# parse one requirement (add --analyze for type/EARS/quality)
python -m reqgraph parse "When a fault is detected, the controller shall isolate the unit within 100 ms." --analyze

# use the shipped, pre-trained BERT tagger
python -m reqgraph parse "The system shall close the valve within 5 seconds." \
        --backend bert --model models/req_tagger --format elements

# train your own (built-in aerospace seed corpus; use bert-base-uncased for production)
python -m reqgraph train --model bert-base-uncased --epochs 80 --out models/req_tagger

# batch a spreadsheet / ReqIF set; quality + duplicate/conflict report
python -m reqgraph batch reqs.csv --out reqs.reqif
python -m reqgraph analyze reqs.csv

# find cross-requirement SUBJECT/OBJECT connections in a set, visualize them
python -m reqgraph connections reqs.csv --format mermaid
```

A ready-to-use tagger trained on the 30-requirement seed corpus
([`reqgraph/seed_data.py`](seed_data.py)) ships in `models/req_tagger`.

## Package layout

```
reqgraph/
  core.py        graph model + exporters (JSON/Mermaid/DOT/Cypher/GraphML/Turtle/networkx)
  templates.py   Template + RUPP + EARS + register_template
  tiling.py      tile_to_graph()  — the lossless engine
  parser.py      RequirementParser + build_requirement
  extractors.py  Extractor ABC + Rule / spaCy / BERT backends + registry
  nlp.py         BertTokenTagger (trainable) + RequirementAnalyzer
  quality.py     IREB quality smells + type + EARS classification
  io_formats.py  CSV / Excel / ReqIF
  corpus.py      requirement-set SUBJECT/OBJECT cross-referencing + connections graph
tests/           pytest suite (lossless round-trip is the headline invariant)
```

## Element roles (IREB / Rupp MASTeR)

`CONDITION` · `SUBJECT` · `MODALITY` (→ legal obligation) · `ACTOR` (Rupp type-2
"whom") · `PROCESS` · `OBJECT` · `DETAILS` · `CONSTRAINT`, joined by logical
`OPERATOR` (AND/OR) nodes for compound actions.

## Limitations (honest)

* **Lossless round-trip is unconditional.** Semantic *labelling* is best-effort.
* The rule backend is most precise on boilerplate; spaCy is better on free prose
  but weaker on the Rupp "provide … with the ability to" form.
* `bert-tiny` is a smoke-test model — embeddings/tagging are weak; use a larger
  checkpoint and real labelled data for production.
* The ReqIF writer is a minimal valid subset (ID + text); extend the attribute
  set for full tool interop.
