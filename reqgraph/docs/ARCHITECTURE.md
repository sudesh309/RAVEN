# reqgraph — Architecture & Data-Flow Reference (v1.0.0)

How data is processed at **module and function level**, from raw requirement
text to graph and back, including the ML, quality, and I/O pipelines.

---

## 1. System overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                            APPLICATION LAYER                           │
│   __main__.py (CLI: parse/batch/train/analyze/gui)   your Python code  │
│   gui.py (local web GUI: stdlib http.server + static/index.html)       │
├────────────────────────────────────────────────────────────────────────┤
│                            SERVICE LAYER                               │
│   parser.RequirementParser      io_formats (CSV/Excel/ReqIF)           │
│   parser.build_requirement      quality.enrich (smells/type/EARS)      │
│                                 nlp.RequirementAnalyzer (dup/conflict) │
├────────────────────────────────────────────────────────────────────────┤
│                      EXTRACTION LAYER (Strategy)                       │
│   extractors.Extractor (ABC)                                           │
│     ├─ RuleExtractor   (deterministic anchors, default)                │
│     ├─ SpacyExtractor  (dependency parse)                              │
│     └─ BertTaggerExtractor ──▶ nlp.BertTokenTagger (PyTorch)           │
├────────────────────────────────────────────────────────────────────────┤
│                        ENGINE LAYER (invariant)                        │
│   tiling.tile_to_graph  — lossless char-level tiling                   │
│   templates.Template    — frozen config (markers, slot order)          │
├────────────────────────────────────────────────────────────────────────┤
│                            MODEL LAYER                                 │
│   core.RequirementGraph / Node / Edge / Role / Rel                     │
│   exporters: JSON · Mermaid · DOT · Cypher · networkx                  │
│   errors.py (exception hierarchy)  ·  logging (per-module loggers)     │
└────────────────────────────────────────────────────────────────────────┘
```

**Dependency rule:** layers depend only downward. The engine and model layers
have **zero third-party dependencies**; spaCy/torch/pandas are imported lazily
inside the functions that need them, so the core works on any machine.

### Core data types

| Type | Shape | Produced by | Consumed by |
|---|---|---|---|
| `text` | `str` | user / file readers | extractors, tiler |
| `Claim` | `(start:int, end:int, Role, attrs:dict)` | extractors | tiler |
| `RequirementGraph` | nodes dict + edges list + `leaf_order` | tiler | exporters, quality, io |
| `leaf_order` | `list[node_id]` in text order | tiler | `generate()` (lossless backbone) |
| BIO labels | `list[int]` per wordpiece token | `BertTokenTagger._encode` | torch training loop |
| embeddings | `float32 [n, hidden]`, L2-normalised | `RequirementAnalyzer.embed` | similarity/dup/conflict |
| DataFrame row | id, text, per-element columns, type, EARS, smells, error | `requirements_to_dataframe` | CSV/Excel writers |

---

## 2. The lossless invariant (why the design is safe)

```
INVARIANT: concat(node.text for node in leaf_order) == original text, always.
```

Enforced in exactly one place — `tiling.tile_to_graph` — by construction:
accepted claims own their spans, every gap becomes a `GLUE` node, nothing is
normalised or rewritten. Consequences:

* any extractor (even a buggy custom one) can only mislabel, never corrupt;
* `_sanitise` clamps out-of-range spans and drops malformed claims before tiling;
* `RequirementGraph.verify()` re-checks the invariant (used by `from_dict`).

---

## 3. Pipeline 1 — single requirement: text → graph → text

```
caller
  │  text:str, metadata:dict
  ▼
parser.RequirementParser.parse                       [parser.py]
  │  1. validate text is str (TypeError otherwise)
  │  2. extractor.extract(text, template) ──▶ list[Claim]
  │  3. tile_to_graph(text, claims, template.name, metadata)
  ▼
tiling.tile_to_graph                                 [tiling.py]
  │  _sanitise(claims, len(text))      clamp/drop bad spans
  │  _dedup_nonoverlap(claims)         sort by (start, -length); greedy accept
  │  tiling loop                       claim spans → element Nodes,
  │                                    gaps → GLUE Nodes, build leaf_order
  │  NEXT chain                        terminal[i] ─NEXT→ terminal[i+1]
  │  semantic tree                     ROOT ─HAS_*→ condition/subject/modality
  │                                    PROCESS starts an ACTION; OBJECT attaches;
  │                                    ≥2 ACTIONs → OPERATOR(AND|OR) via
  │                                    _infer_operator(gap text)
  │                                    actor/details/constraint → head ACTION
  ▼
core.RequirementGraph
  │  generate()  = "".join(leaf text)   ←──  lossless inverse
  │  elements() / by_role() / summary()
  │  to_json/to_mermaid/to_dot/to_cypher/to_networkx
  ▼
caller
```

### 3a. RuleExtractor.extract — stage-by-stage window narrowing

Input: full text. Each stage consumes an anchor and narrows the window the next
stage searches. All regexes come precompiled from `_patterns(template)`
(`functools.lru_cache`; `Template` is frozen/hashable).

| # | Stage | Window | Emits |
|---|---|---|---|
| 1 | leading condition (`cond_lead`), terminated at the *last comma before the modality* (protects "14,000") or a `then` | start of text | `CONDITION {marker, kind=pre}` |
| 2 | modality keyword (`modality`); text between condition and modality | after condition | `MODALITY {obligation}`, `SUBJECT` |
| 3 | trailing condition (`cond`) | after modality | `CONDITION {kind=post}`, shrinks window |
| 4 | trailing constraint (`constraint`) | after modality | `CONSTRAINT {marker}`, shrinks window |
| 5 | function type: `ui` ("provide X with the ability to") → `ACTOR`; else `interface` ("be able to") | action window | `ACTOR`, sets `function_type` |
| 6 | `_parse_action`: top-level conjunction followed by a non-determiner word splits two actions; first determiner splits PROCESS/OBJECT | remaining core | `PROCESS {function_type}`, `OBJECT` per action |

### 3b. SpacyExtractor.extract — dependency-tree mapping

```
text ─▶ spaCy pipeline (cached singleton per model name)
     ─▶ Doc(tokens, deps)
        advcl subtree                → CONDITION (pre/post by position vs ROOT)
        prep ∈ constraint_markers    → CONSTRAINT (subtree span)
        "provide…ability to" regex   → ACTOR  (+ function_type)
        ROOT.children aux(MD)        → MODALITY
        ROOT.children nsubj subtree  → SUBJECT (clipped against excl spans)
        ROOT + conj verbs (+prt)     → PROCESS per verb
        verb.children dobj/attr      → OBJECT;  dative → ACTOR
     ─▶ list[Claim]    (no ROOT found ⇒ fall back to RuleExtractor)
```

### 3c. BertTaggerExtractor.extract

```
text ─▶ BertTokenTagger.predict_spans
          tokenizer(offsets, truncation@max_len → warning if hit)
          model logits [1, T, |labels|] ─argmax─▶ label ids
          BIO decode over offset mapping ─▶ [(start, end, Role)]
     ─▶ wrap as Claims (+obligation for MODALITY, function_type for PROCESS)
```

---

## 4. Pipeline 2 — authoring: elements → text → graph

```
build_requirement({Role: text}, template)            [parser.py]
  │  normalise keys (Role or "NAME")
  │  lay out in template.slot_order; CONDITION gets a trailing comma;
  │  final period appended if missing
  │  re-parse the rendered text  →  ordinary RequirementGraph
  ▼
RequirementGraph (round-trips by construction)
```

---

## 5. Pipeline 3 — BERT training & inference         [nlp.py]

```
TRAIN
seed_data.training_examples() | JSONL loader
  │  [(text, [(s,e,Role)...]), ...]
  ▼
BertTokenTagger.train
  │  _validate_examples       bounds + types (fail fast, clear message)
  │  torch.manual_seed(seed)  reproducible runs (default seed=42)
  │  _encode per example      tokenizer(offset_mapping)
  │                           char span lookup per token → B-/I-/O label ids
  │                           special tokens → -100 (ignored by loss)
  │  TensorDataset → DataLoader(shuffle) → AdamW
  │  epochs × (forward → CrossEntropy(loss) → backward → step)
  ▼
model.eval() ── save_pretrained(dir) ⇄ BertTokenTagger.load(dir)

EVALUATE (quality gate)
  evaluate(examples) → exact-span {precision, recall, f1, gold_spans}

INFER
  predict_spans(text) → [(s,e,Role)]  (see 3c)
```

Checkpoint quirks handled: `BertConfig` built explicitly (tiny checkpoints omit
`model_type`); tokenizer defaults to `bert-base-uncased` fast tokenizer (tiny
checkpoints ship none; same WordPiece vocab).

---

## 6. Pipeline 4 — analysis                          [nlp.py · quality.py]

```
RequirementAnalyzer.embed(texts)
  tokenizer(batch, padding) → BERT last_hidden_state [n,T,H]
  mask-weighted mean over T → [n,H] → L2 normalise
similarity(a,b)        = e_a · e_b (cosine)
find_duplicates(reqs)  = pairs with sim ≥ threshold (sorted desc)
detect_conflicts(reqs) = pairs with sim ≥ threshold AND opposite polarity
                         (negation regex on each text)

quality.enrich(graph)                       writes graph.analysis
  check_quality(text)    precompiled regexes: weak words, passive voice,
                         missing modality, vague quantifier, non-atomic
  classify_type(text)    keyword scoring → functional/performance/interface/
                         safety/usability
  classify_ears(graph)   CONDITION markers → ubiquitous/event/state/unwanted/
                         optional
```

---

## 7. Pipeline 5 — batch I/O                         [io_formats.py]

```
read_requirements_csv/excel(path)           read_reqif(path)
  pandas.read_* → _rows_from_df               lxml parse → SPEC-OBJECT walk
  (validates text column, skips blank         → [(id, text)]
   rows, cleans NaN ids) → [(id, text)]
            │
            ▼
requirements_to_dataframe(items, template, extractor)
  per row: parse → enrich → element buckets → flat row
  FAULT ISOLATION: a failing row records its error in an "error"
  column instead of aborting the batch
            │
            ▼
write_csv / write_excel (pandas, openpyxl)   write_reqif (lxml: header,
                                             datatype, spec-type, SPEC-OBJECTs)
```

---

## 8. CLI data flow                                  [__main__.py]

```
argv ─▶ argparse (global: -v/-vv → logging level, --version)
  parse    text ─▶ _build_parser(template, backend, model) ─▶ parse [+enrich]
           ─▶ format switch: summary|json|mermaid|dot|cypher|elements ─▶ stdout
  batch    infile ─▶ _read_items (by extension) ─▶ dataframe/writers ─▶ out
  train    seed corpus | JSONL ─▶ BertTokenTagger.train ─▶ save dir
  analyze  infile ─▶ per-row enrich report + embed dup/conflict report
errors: any ReqGraphError ─▶ exit("error: …")  (no tracebacks for users)
```

---

## 8a. Pipeline 6 — GUI                              [gui.py · static/index.html]

```
browser ──GET /──▶ static/index.html  (single page, no CDN, offline)
        ──GET /api/info──▶ {templates, backends availability, version}
        ──POST /api/parse {text, template, backend}──▶
gui.parse_request (pure function)
  │  GuiState.extractor(name)     lazy, lock-guarded backend cache
  │  RequirementParser.parse      (pipeline 1) + quality.enrich
  │  _compute_kpis                round-trip, score 0-100 (smell deductions),
  │                               coverage % = semantic chars / total chars,
  │                               type, EARS, obligation, actions/operator
  │  _graph_to_tree               nested ROOT-down tree, children in text order
  │  tiles                        leaf_order → [{role, text}] (lossless strip)
  ▼
JSON ──▶ browser JS renders: KPI cards · tiled text · SVG tree layout
         (leaf-slot tidy layout, role color map) · element table · downloads
Security: server binds 127.0.0.1 only; errors → {error} JSON, never tracebacks.
```

---

## 9. Performance & resource profile (individual computer)

| Path | Cost | Mechanism |
|---|---|---|
| Rule parse | **~0.05 ms/req (~22 k req/s)** | `_patterns` lru_cache: zero regex recompilation after first parse per template |
| spaCy parse | ~5–15 ms/req | pipeline loaded once per model (`_nlp_cache` singleton) |
| BERT tag (bert-tiny, CPU) | ~10–30 ms/req | model lazy-loaded once; GPU used automatically if available |
| Embedding batch | one forward pass per batch | mean-pool, vectorised numpy cosine matrix |
| Memory | core: trivial; spaCy ~80 MB; bert-tiny ~70 MB RAM | all lazy — you pay only for what you import |

Offline use: after the first download, set `HF_HUB_OFFLINE=1` and everything
(spaCy model, HF checkpoints in `~/.cache/huggingface`) runs without internet.

---

## 10. Error handling & logging

```
errors.ReqGraphError
  ├─ GraphIntegrityError   broken tiling / dangling edge (core.verify, add_edge)
  ├─ ExtractionError       backend unavailable/misconfigured (extractors)
  ├─ TemplateError         malformed/unknown template
  ├─ DataFormatError       bad file/payload shape (io_formats, from_dict)
  └─ ModelError            ML model load/train/apply failures
```

* Each module logs to `logging.getLogger(__name__)`; the package installs a
  `NullHandler` (library etiquette — silent unless the app configures logging).
* The CLI maps `-v/-vv` to INFO/DEBUG and converts `ReqGraphError` to a clean
  one-line exit message.
* Defensive boundaries: claims are sanitised before tiling; `from_dict` verifies
  integrity; batch rows are fault-isolated; BERT truncation warns instead of
  silently dropping.

---

## 11. Extension points

| You want | You touch | Everything else unchanged |
|---|---|---|
| new requirement style | `Template(...)` + `register_template` | parser, tiler, exporters |
| new extraction logic | subclass `Extractor`, return Claims | losslessness guaranteed by tiler |
| new element role | `core.Role` (+ `TERMINAL_ROLES`), emit it from an extractor | tiling keeps working |
| new export format | method on `RequirementGraph` | — |
| better ML | swap `model_name` (e.g. `bert-base-uncased`), retrain | same train/save/load API |
