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

## GUI

```bash
python -m reqgraph gui          # opens http://127.0.0.1:8765 in your browser
```

A zero-dependency local web app (stdlib `http.server`, bound to 127.0.0.1, fully
offline): type or pick a requirement, switch template/backend, and see
**KPI cards** (round-trip, quality score, elements/coverage, type, EARS,
obligation, parse time), the **color-tiled requirement** (every character mapped
to its owning node), the **semantic graph** rendered as an SVG tree, and the
**element table** — with one-click SVG/JSON/Mermaid/DOT export.

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
```

A ready-to-use tagger trained on the 30-requirement seed corpus
([`reqgraph/seed_data.py`](seed_data.py)) ships in `models/req_tagger`.

## Package layout

```
reqgraph/
  core.py        graph model + exporters (JSON/Mermaid/DOT/Cypher/networkx)
  templates.py   Template + RUPP + EARS + register_template
  tiling.py      tile_to_graph()  — the lossless engine
  parser.py      RequirementParser + build_requirement
  extractors.py  Extractor ABC + Rule / spaCy / BERT backends + registry
  nlp.py         BertTokenTagger (trainable) + RequirementAnalyzer
  quality.py     IREB quality smells + type + EARS classification
  io_formats.py  CSV / Excel / ReqIF
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
