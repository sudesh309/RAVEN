# reqgraph — Architecture & Data-Flow Reference (v2.0.0)

How data is processed at **module and function level**, from raw requirement
text to graph and back, including the ML, quality, I/O, and model-comparison
pipelines.

---

## 1. System overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                             APPLICATION LAYER                              │
│  __main__.py  CLI: parse / batch / train / analyze / connections /         │
│               export / compare / compare-v1 / gui                          │
│  gui.py       local web GUI  (stdlib http.server + static/index.html)      │
├────────────────────────────────────────────────────────────────────────────┤
│                          MODEL COMPARISON LAYER                            │
│  sysml_parser.py     SysML v2 textual notation → SysMLModel                │
│  sysml_compare.py    role-bucketed comparison  → ComparisonReport          │
│  sysml_v1_parser.py  SysML v1 XMI / Turtle    → SysMLV1Model + KG         │
│  sysml_v1_compare.py context-aware comparison  → V1ComparisonReport        │
│                       BFS neighborhood scoring · OntologyDiff              │
├────────────────────────────────────────────────────────────────────────────┤
│                             SERVICE LAYER                                  │
│  parser.RequirementParser        io_formats (CSV/Excel/JSON/ReqIF)         │
│  parser.build_requirement        quality.enrich (smells/type/EARS)         │
│                                  nlp.RequirementAnalyzer (dup/conflict)    │
│  corpus.build_requirement_set_graph                                        │
│    to_element_graphml()  to_req_turtle()  to_turtle()  to_cypher() …       │
├────────────────────────────────────────────────────────────────────────────┤
│                        EXTRACTION LAYER (Strategy)                         │
│  extractors.Extractor (ABC)                                                │
│    ├─ RuleExtractor   (deterministic anchors, default)                     │
│    ├─ SpacyExtractor  (dependency parse)                                   │
│    └─ BertTaggerExtractor ──▶ nlp.BertTokenTagger (PyTorch)                │
├────────────────────────────────────────────────────────────────────────────┤
│                          ENGINE LAYER (invariant)                          │
│  tiling.tile_to_graph  — lossless char-level tiling                        │
│  templates.Template    — frozen config (markers, slot order)               │
├────────────────────────────────────────────────────────────────────────────┤
│                             MODEL LAYER                                    │
│  core.RequirementGraph / Node / Edge / Role / Rel                          │
│  exporters: JSON · Mermaid · DOT · Cypher · GraphML · Turtle · networkx    │
│  errors.py (exception hierarchy)  ·  logging (per-module loggers)         │
└────────────────────────────────────────────────────────────────────────────┘
```

**Dependency rule:** layers depend only downward. The engine and model layers
have **zero third-party dependencies**; spaCy/torch/pandas/rdflib are imported
lazily inside the functions that need them, so the core works on any machine.

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

## 7a. Pipeline 5a — requirement-set connections      [corpus.py]

```
build_requirement_set_graph(items, roles=(SUBJECT,OBJECT), threshold, similarity)
  items ─▶ _normalise_items → (id, text) pairs
  per item: parser.split(text) → parser.parse(seg) for each segment
            (compound items become REQ-i-1, REQ-i-2, ...; pipeline 1 reused)
            ─▶ collect ElementRef(req_id, node_id, role, text) for each
              SUBJECT/OBJECT (or chosen roles) element across ALL requirements
            │
            ▼
  _score_fn(elements, similarity)
    "lexical"   token-Jaccard ⊔ SequenceMatcher.ratio, stopwords stripped
                (zero dependencies, deterministic)
    "embedding" RequirementAnalyzer.embed(texts) → cosine similarity matrix
            │
            ▼
  pairwise compare elements from DIFFERENT requirements, same role;
  score ≥ threshold ─▶ Connection(a, b, role, score)
            │
            ▼
  RequirementSetGraph(req_ids, texts, graphs, connections)
    .to_mermaid()/.to_dot()/.to_graphml()/.to_turtle()/.to_cypher()/.to_dict()
    — one node per requirement, one edge per (req_a, req_b, role) connection
```

---

## 8. CLI data flow                                  [__main__.py]

```
argv ─▶ argparse (global: -v/-vv → logging level, --version)
  parse        text ─▶ _build_parser(template, backend, model) ─▶ parse [+enrich]
               ─▶ format switch: summary|json|mermaid|dot|cypher|elements ─▶ stdout
  batch        infile ─▶ _read_items (by extension) ─▶ dataframe/writers ─▶ out
  train        seed corpus | JSONL ─▶ BertTokenTagger.train ─▶ save dir
  analyze      infile ─▶ per-row enrich report + embed dup/conflict report
  connections  infile ─▶ build_requirement_set_graph ─▶ text|json|mermaid|dot|
               graphml|turtle|cypher (pipeline 5a)
  export       infile (CSV/Excel/JSON/ReqIF) ─▶ parse+enrich+connections ─▶
               --csv / --json / --graphml / --req-turtle  (pipeline 6a)
  compare      model.sysml reqs ─▶ parse_sysml + compare ─▶ text|json
               --report PATH  --graphml PATH  (pipeline 7)
  compare-v1   model.xmi|ttl reqs ─▶ read_sysml_v1 + compare_v1 ─▶ text|json
               --context-hops N  --kg PATH  --out-turtle PATH
               --ontology-graphml PATH  (pipeline 8)
errors: any ReqGraphError ─▶ exit("error: …")  (no tracebacks for users)
```

---

## 8a. Pipeline 6 — GUI                              [gui.py · static/index.html]

Five endpoints, all pure functions (no side-effects beyond the backend cache):

```
browser ──GET /──▶ static/index.html  (single page, no CDN, offline)
        ──GET /api/info──▶ {templates, backends availability, version}

──POST /api/parse {text, template, backend}──▶
  gui.parse_request
    │ parse (pipeline 1) + enrich + _compute_kpis + _graph_to_tree + tiles
    ▼ JSON: KPI cards · tiled text · SVG tree · element table · downloads

──POST /api/connections {texts, similarity, threshold, roles, …}──▶
  gui.connections_request ─▶ build_requirement_set_graph (pipeline 5a)
    ▼ JSON: {requirements, connections, mermaid, dot, graphml, turtle, cypher}
      circular network graph · connections table · downloads

──POST /api/export {content, format, threshold, similarity, …}──▶
  gui.export_request ─▶ reader → build_requirement_set_graph → enrich →
    requirements_to_dataframe  (pipeline 6a)
    ▼ JSON: {requirements, connections, n_*, mermaid, graphml,
             turtle, req_turtle, cypher, csv_data}
      quality table (auto-discovers extra metadata columns) · connections
      graph · download buttons (CSV / JSON / GraphML / Req Turtle / …)

──POST /api/compare {model_content, req_content, threshold, similarity, …}──▶
  gui.compare_request ─▶ parse_sysml + compare  (pipeline 7)
    ▼ JSON: {semantic_match, model_coverage, req_coverage,
             role_breakdown, matches, unmatched_model, unmatched_reqs,
             graphml, mermaid, warnings}
      KPI tiles (colour-coded) · role breakdown · match table · gap panels

──POST /api/compare-v1 {model_content, model_format, req_content, …}──▶
  gui.compare_v1_request ─▶ parse_sysml_v1 + compare_v1  (pipeline 8)
    ▼ JSON: {semantic_match, model_coverage, req_coverage, role_breakdown,
             matches (with name/context/satisfaction scores), unmatched_*,
             graphml, mermaid, kg_graphml, model_turtle,
             ontology_mermaid, ontology_graphml, warnings}
      confidence decomposition table · Ontology View tab · downloads

Security: server binds 127.0.0.1 only; errors → {error} JSON, never tracebacks.
GuiState: thread-safe extractor cache (one per backend); shared across requests.
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
| new cross-requirement similarity metric | `_score_fn` in `corpus.py` | `build_requirement_set_graph`, CLI/GUI unchanged |
| new SysML v2 keyword → role | `_KEYWORD_ROLE` dict in `sysml_parser.py` | comparison engine unchanged |
| new SysML v1 stereotype → role | `_STEREO_ROLE` / `_UML_ROLE` dicts in `sysml_v1_parser.py` | parser, KG, comparison unchanged |
| different context scoring formula | `compare_v1()` weights in `sysml_v1_compare.py` | parser, KG export unchanged |
| additional model input format | add a parser branch in `read_sysml_v1()` | comparison engine unchanged |

---

## 12. Pipeline 7 — SysML v2 comparison              [sysml_parser.py · sysml_compare.py]

```
model_text (SysML v2 textual notation)
  ▼
sysml_parser.parse_sysml(text)                       [sysml_parser.py]
  │  Strip // line comments; extract doc /* … */ blocks
  │  _DECL_RE regex: match (keyword, name) pairs across all SysML keywords
  │  Track package nesting via brace depth
  │  Map keyword → IREB Role:
  │    part_def / part          → SUBJECT
  │    action_def / action      → PROCESS
  │    attribute_def / attribute → OBJECT
  │    state_def / state / transition → CONDITION
  │    port_def / port          → ACTOR
  │    require/assume constraint → CONSTRAINT
  │    requirement{}            → extracted to sysml_requirements list
  ▼
SysMLModel(elements: list[SysMLElement], sysml_requirements, packages, source_path)

sysml_compare.compare(model, items, roles, threshold, similarity, …)
  │  build_requirement_set_graph(items, threshold=0.0) → requirement elements
  │  Per role R:
  │    model_R  = [e.name for e in model.elements if e.role == R]
  │    req_R    = [(ref.text, req_id) for ref in req_elements if ref.role == R]
  │    score matrix (|model_R| × |req_R|) via lexical or embedding scorer
  │    matched_model_R: best row score ≥ threshold
  │    matched_req_R:   best col score ≥ threshold
  │    role_score_R = harmonic_mean(model_cov_R, req_cov_R)
  │  semantic_match = weighted_avg(role_score_R) by max(|model_R|,|req_R|)
  ▼
ComparisonReport
  .model_coverage  .req_coverage  .semantic_match
  .role_breakdown  .matches (MatchDetail)
  .unmatched_model .unmatched_reqs  .warnings
  .to_graphml()    bipartite: MODEL nodes ↔ REQ_ELEMENT nodes, MATCHES edges
  .to_mermaid()    flowchart: element → matched requirement text
  .to_dict()       JSON-serialisable
```

---

## 13. Pipeline 8 — SysML v1 KG + context-aware comparison
##                              [sysml_v1_parser.py · sysml_v1_compare.py]

### 13a. Parsing (two formats)

```
read_sysml_v1(path)  ─▶  auto-detect by extension + content sniff
  │  .ttl / .rdf / .n3 / .owl  or  @prefix / @base keyword  → Turtle path
  │  .xmi / .uml / .xml        or  <?xml / <uml:             → XMI path
  ▼

XMI path  ─▶  _parse_xmi(text, source_path)
  PASS 1: xml.etree.ElementTree walk
    ├─ collect every element by xmi:id into elem_map {id → ET.Element}
    ├─ record packagedElement relationships with package hierarchy
    └─ note root-level siblings (stereotype applications)
  PASS 2: resolve
    ├─ stereotype applications (SysML:Block, SysML:Requirement, SysML:Satisfy…)
    │   matched to elements via base_Class / base_Abstraction / base_Property
    │   _is_sysml_ns() accepts any known SysML namespace URI regardless of prefix
    ├─ Abstraction elements: client xmi:idref + supplier xmi:idref → V1Relation
    │   rel_type from stereotype (satisfy/refine/derive/trace/allocate)
    ├─ ownedAttribute aggregation="composite" → composition V1Relation
    ├─ ownedComment body attribute → V1Element.doc
    ├─ UML type + stereotype → IREB Role  (_UML_ROLE / _STEREO_ROLE tables)
    └─ SysML:Requirement text= / id= attributes → V1Element.req_text / .req_id

Turtle path  ─▶  _parse_turtle(text, source_path)   (requires rdflib≥6.0)
  rdflib.Graph.parse(data=text, format="turtle")
  For each subject:
    collect rdf:type(s) → local name → Role via _TTL_TYPE_ROLE table
    rdfs:label → name;  rdfs:comment / sysml:text → doc / req_text
    sysml:id → req_id
    predicate local name in RELATION_PREDS → V1Relation
  Custom subclass support: rdfs:subClassOf chain → inherits Role from ancestor
  element.xmi_id = URI fragment or local name (so adjacency indexing works)

Both paths produce:
  SysMLV1Model(elements: list[V1Element], relations: list[V1Relation], …)
    .to_graphml()  full KG GraphML (MODEL_ELEMENT + REQUIREMENT nodes, typed edges)
    .to_turtle()   canonical Turtle using sysmlkg: ontology (round-trippable)
```

### 13b. Context building and comparison

```
compare_v1(model, items, roles, threshold=0.5, context_hops=2, …)
  │
  │  build_requirement_set_graph(items, threshold=0.0) → req_elements
  │
  │  For each model element e:
  │    _build_context(e, model, hops)        [BFS over V1Relation adjacency]
  │      parts = [e.name, e.doc, e.req_text]
  │      hop 1: all neighbors via any relation → append neighbor.name + .doc
  │      hop 2: neighbors of neighbors → append their name + .doc
  │      context_string = " ".join(non-empty parts)
  │
  │  For each (model element e, req element r) pair of same role:
  │    name_score    = _score(e.name, r.text)         lexical or embedding
  │    context_score = _score(context_string, r.text)
  │    sat_bonus     = 1.0 if model has satisfy/refine link from e to a
  │                    model requirement whose req_text scores ≥ 0.5 vs r.text
  │                    else 0.0
  │    confidence    = 0.25·name_score + 0.55·context_score + 0.20·sat_bonus
  │
  │  Best-match per element; matched if confidence ≥ threshold
  │  model_coverage, req_coverage, semantic_match (F1)
  │  ontology_diff(model, rsg) → OntologyDiff
  ▼
V1ComparisonReport
  .model_coverage  .req_coverage  .semantic_match
  .role_breakdown  .matches (V1MatchDetail: name_score, context_score,
  .unmatched_*     confidence, context_used, hops_used, via_satisfaction)
  .to_graphml()    bipartite + CONTEXT_HOP_1/2 edges
  .to_mermaid()    flowchart
  .to_dict()       JSON-serialisable (all score components exposed)
```

### 13c. Ontology diff visualisation

```
ontology_diff(model: SysMLV1Model, rsg: RequirementSetGraph) → OntologyDiff

  Requirement ontology graph (from RSG):
    Nodes = IREB role classes with instance counts
    Edges = structural role relations (HAS_SUBJECT, HAS_PROCESS, …)

  Model ontology graph (from SysMLV1Model):
    Nodes = SysML element type+stereotype classes with instance counts
    Edges = relation types (composition, association, satisfy, refine, …)

  Mapping layer:
    Cross-graph MAPS_TO edges: model type → req role, labelled mean confidence

OntologyDiff
  .req_ontology_nodes / _edges   .model_ontology_nodes / _edges   .mappings
  .to_mermaid()   two LR subgraphs (reqont / modelont) + dashed MAPS_TO edges
  .to_graphml()   combined bipartite ontology graph
  .to_dict()      JSON-serialisable
```

---

## 14. Pipeline 6a — Extended export                 [corpus.py · io_formats.py]

```
All import readers return 3-tuples: (id: str|None, text: str, meta: dict)
  read_requirements_csv / read_requirements_excel
    _rows_from_df: extra columns → normalised names (lower, spaces→_) → meta dict
  read_requirements_json
    array-of-objects, dict-keyed, or plain-string-dict shapes
    normalised through pandas DataFrame → _rows_from_df
  read_reqif
    all ATTRIBUTE-VALUE-STRING nodes → meta dict (not just id/text)

_normalise_items(items): yields (id, text, meta) 3-tuples
  consumed by build_requirement_set_graph and requirements_to_dataframe

requirements_to_dataframe(items, template, extractor)
  extra meta columns injected between text and type (order: first-seen)
  output columns: id · text · [meta…] · type · ears_pattern · condition ·
                  subject · modality · actor · process · object · constraint ·
                  weak_words · non_atomic · roundtrip_ok · error

RequirementSetGraph new methods:
  .to_element_graphml()
    REQ nodes:     node_type=REQ, text, quality attrs, metadata JSON
    ELEMENT nodes: node_type=ELEMENT, role, text  (id = req_id__node_id)
    HAS_ELEMENT edges:   REQ → each of its ELEMENT nodes
    SIMILAR_* edges:     ELEMENT ↔ ELEMENT across requirements (from connections)
    visualisable in Gephi / yEd / Cytoscape as a full semantic knowledge graph

  .to_req_turtle()
    Serialises requirement set as RDF ontology (reqont: namespace)
    :R1 a reqont:Requirement ; reqont:id "R1" ; reqont:text "…" .
    :R1_subject a reqont:Subject ; rdfs:label "vehicle" ; reqont:fromRequirement :R1 .
    Cross-req similarity: :R1_subject reqont:similarTo :R2_subject .
    Output is rdflib-parseable and loadable into a triplestore alongside model TTL
```

---

## 15. Module inventory (complete)

| Module | Layer | Key public API |
|---|---|---|
| `core.py` | Model | `Role`, `Rel`, `Node`, `Edge`, `RequirementGraph` |
| `errors.py` | Model | `ReqGraphError` hierarchy |
| `templates.py` | Engine | `Template`, `RUPP_TEMPLATE`, `EARS_TEMPLATE`, `register_template` |
| `tiling.py` | Engine | `tile_to_graph` |
| `extractors.py` | Extraction | `RuleExtractor`, `SpacyExtractor`, `BertTaggerExtractor`, `get_extractor` |
| `parser.py` | Service | `RequirementParser`, `build_requirement`, `split_requirements` |
| `nlp.py` | Service | `BertTokenTagger`, `RequirementAnalyzer` |
| `quality.py` | Service | `enrich`, `check_quality`, `classify_type`, `classify_ears` |
| `io_formats.py` | I/O | `read_requirements_csv/excel/json`, `read_reqif`, `write_reqif`, `requirements_to_dataframe` |
| `corpus.py` | Service | `build_requirement_set_graph`, `RequirementSetGraph` |
| `seed_data.py` | Data | 30-requirement aerospace seed corpus |
| `sysml_parser.py` | Comparison | `SysMLElement`, `SysMLModel`, `parse_sysml`, `read_sysml` |
| `sysml_compare.py` | Comparison | `MatchDetail`, `ComparisonReport`, `compare` |
| `sysml_v1_parser.py` | Comparison | `V1Element`, `V1Relation`, `SysMLV1Model`, `parse_sysml_v1`, `read_sysml_v1` |
| `sysml_v1_compare.py` | Comparison | `V1MatchDetail`, `V1ComparisonReport`, `OntologyDiff`, `compare_v1`, `ontology_diff` |
| `traceability.py` | Comparison | `TraceabilityMatrix`, `TraceItem`, `Finding`, `VerificationMethod`, `TraceStatus`, `build_traceability_matrix` — the RVTM: verified (auditable model link) vs candidate (semantic) vs gap, IADT verification method, certification findings |
| `__main__.py` | Application | CLI subcommands |
| `gui.py` | Application | HTTP server, 5 request handlers |
| `static/index.html` | Application | Single-page GUI (no CDN, offline-capable) |
