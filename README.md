# RAVEN — Requirement Analysis and Visualisation Engine

RAVEN checks, connects and formalises system requirements. It parses each
requirement into its IREB roles (subject, condition, action, object…), scores its
quality and completeness, traces it against SysML models, and can turn it into a
model-checkable TLA+ specification. It runs **locally and offline** by default;
the AI assistant is optional and can stay on your machine (Ollama) or use Google
Cloud (Vertex AI).

This guide covers **installing, configuring and using every function**. For the
Python API and internals, see the [developer reference](reqgraph/README.md).

| Function | Use it for | GUI | CLI |
|---|---|---|---|
| [Parse & analyse](#31-parse-and-analyse-one-requirement) | quality, completeness, type and EARS pattern of one requirement | card **1** | `parse` |
| [Connections](#32-find-connections-in-a-set) | related and duplicate requirements in a set | card **2** | `connections` |
| [Import & analyse](#33-import-and-analyse-a-document) | a whole Word / ReqIF / JSON / CSV / Excel document: quality, completeness, types, traceability | card **3** | `analyze`, `export`, `batch` |
| [Compare SysML v2](#34-compare-a-sysml-v2-model) | how well a SysML v2 model covers the requirements | card **4** | `compare` |
| [Compare SysML v1 + RVTM](#35-compare-a-sysml-v1--cameo-model-and-build-the-rvtm) | Cameo `.mdzip` / XMI / Turtle models; the certification traceability matrix | card **5** | `compare-v1` |
| [PlusCal / TLA+](#36-formalise-as-pluscal--tla) | a formal, model-checkable spec of a requirement | card **6** | `pluscal` |
| [AI assistant](#37-ai-assistant-rewrite-suggestions-and-pluscal-refinement) | rewrite suggestions; refining PlusCal models | **⚙ AI assistant** | `rewrite`, `llm-status` |

---

## Contents

1. [Install](#1-install)
2. [Start the GUI](#2-start-the-gui)
3. [Use each function](#3-use-each-function)
4. [Configure each component](#4-configure-each-component)
5. [Environment variable reference](#5-environment-variable-reference)
6. [Input file formats](#6-input-file-formats)
7. [Troubleshooting](#7-troubleshooting)
8. [Tests and further documentation](#8-tests-and-further-documentation)

---

## 1. Install

**You need:** Python 3.10 or newer. Everything else is optional and only needed
for the feature that uses it.

### Windows — quickest start

Double-click **`launch_raven.bat`** in the project folder. It installs the file
readers (CSV, Excel, ReqIF) and opens the GUI in your browser. Add further
features with the extras below.

### Any platform — with pip

From the project folder:

```bash
pip install -e .              # core: parsing, quality, connections, SysML, PlusCal, AI assistant
pip install -e ".[io]"        # + CSV / Excel / JSON / ReqIF import          (recommended)
```

Add the extras for the features you want:

| Extra | Command | Unlocks |
|---|---|---|
| `io` | `pip install -e ".[io]"` | CSV, Excel, JSON and ReqIF import/export (pandas, openpyxl, lxml) |
| `ml` | `pip install -e ".[ml]"` | BERT backend and **BERT embedding** similarity (torch, transformers) — several GB |
| `nlp` | `pip install -e ".[nlp]"` | spaCy backend (then download a model, see [§4.1](#41-extraction-backends)) |
| `graph` | `pip install -e ".[graph]"` | networkx export |
| `gcp` | `pip install -e ".[gcp]"` | Vertex AI with a service-account key (optional — `gcloud` works without it) |
| `all` | `pip install -e ".[all]"` | io + ml + nlp + graph |
| `dev` | `pip install -e ".[dev]"` | pytest |

Several extras at once: `pip install -e ".[io,ml,nlp]"`.

**Needs nothing extra:** Word (`.docx`) import, the Ollama connection, PlusCal
generation. (Model-checking PlusCal needs Java — see [§4.4](#44-tla-tools-model-checking-pluscal).)

### Check the install

```bash
python -m reqgraph --version
python -m reqgraph llm-status      # which optional AI / TLA+ components are ready
```

---

## 2. Start the GUI

```bash
python -m reqgraph gui
```

This opens <http://127.0.0.1:8765>. Every card loads a worked example, so each
function shows real output immediately.

| Option | Effect |
|---|---|
| `--port 9000` | use another port (if 8765 is taken) |
| `--no-browser` | don't open the browser automatically |
| `--model PATH` | BERT tagger directory for the *bert* backend (default `models/req_tagger`) |

The GUI binds to `127.0.0.1` only — it is not reachable from other machines —
and accepts requests only from its own page.

The bar at the top jumps to each card. **Constraints & roadmap →** (top right)
lists what each function can and cannot do yet.

> **Start RAVEN from the project folder.** Relative defaults such as
> `models/req_tagger` (the shipped BERT tagger) are resolved from the folder you
> launch in.

---

## 3. Use each function

### 3.1 Parse and analyse one requirement

**GUI — card 1.** Type a requirement, choose a **Template** and **Backend**, then
**Parse → graph** (or Ctrl+Enter). You get:

- **KPI tiles** — round-trip check, quality score, **completeness**, type
  (functional / performance / interface / safety / usability), EARS pattern,
  obligation (shall / should / may).
- **Tiled elements** — every character assigned to a role; the graph regenerates
  the exact text.
- **Semantic graph** — download as SVG, JSON, Mermaid, DOT, GraphML, Turtle, Cypher.

What the scores mean:

| Check | Flags | Severity |
|---|---|---|
| Quality | weak words (*fast, user-friendly, optimal…*), passive voice, missing modality, vague quantifiers, non-atomic / compound requirements | −20 per smell, −10 per weak word |
| Completeness (INCOSE C4) | no subject / modality / action; placeholders (`TBD`, `TBC`, `<value>`) | **blocker** |
| | pronoun subject (*"It shall…"*), performance claim without a number, number without a unit, truncated text | **major** |
| | exact value without a tolerance, one-word action | minor |

**CLI:**

```bash
python -m reqgraph parse "When the cabin altitude exceeds 14,000 feet, the oxygen system shall deploy the passenger oxygen masks within 4 seconds." --analyze
python -m reqgraph parse "The pump shall start." --format json      # also: mermaid, dot, graphml, turtle, cypher, elements
```

### 3.2 Find connections in a set

**GUI — card 2.** Paste one requirement per line. Requirements whose subjects or
objects are similar are linked in an interactive graph (drag to pin, scroll to
zoom) and listed in a table.

- **Similarity** — *lexical* (word overlap, no dependencies) or *BERT embedding*
  (catches synonyms like *decelerate ↔ slow down*; needs the `ml` extra —
  [§4.3](#43-bert-embedding-similarity)).
- **Threshold** — 0 to 1. Lower finds more (and weaker) links; 0.3–0.6 is typical.

**CLI:**

```bash
python -m reqgraph connections reqs.csv --threshold 0.5 --format mermaid
python -m reqgraph connections reqs.csv --similarity embedding --roles SUBJECT,OBJECT,PROCESS
```

### 3.3 Import and analyse a document

**GUI — card 3.** Choose **Upload file** (or paste text) and pick the file. RAVEN
reads the requirements and reports, per set:

- **Completeness** — % ready for design; *blocked* / *gaps* counts; the recurring
  findings, blockers first, with the requirement IDs behind each.
- **Traceability & type mix** — % of requirements with an identifier; declared
  parent / satisfies links and whether they resolve inside the document; dangling
  references; duplicate IDs; how many requirements name a verification method;
  the mix of requirement types and EARS patterns.
- **Table** — one row per requirement with its roles, type, weak words,
  completeness verdict and what is missing.
- **Downloads** — quality CSV, JSON, GraphML, Turtle, Cypher.

Accepted files: **Word `.docx`**, **ReqIF** (`.reqif`, `.xml`), **JSON**, **CSV**,
**Excel** (`.xlsx`, `.xls`), plain text. Exports from DOORS, Polarion or Jama load
without renaming columns — see [§6](#6-input-file-formats) for exactly what is
recognised.

> Word files must be **uploaded**; they cannot be pasted (`.docx` is a binary
> format).

**CLI:**

```bash
python -m reqgraph analyze spec.docx                # quality, completeness, type mix, traceability
python -m reqgraph export  reqs.reqif --out-prefix build/out   # build/out.csv .json .graphml .req.ttl
python -m reqgraph batch   reqs.csv --out reqs.reqif           # convert between formats
```

### 3.4 Compare a SysML v2 model

**GUI — card 4.** Paste (or load the example) SysML v2 textual notation and the
requirements. RAVEN matches model elements (parts, actions, states, attributes)
to requirement roles and reports the **semantic match %**, **model coverage**
(elements traced to a requirement), **requirement coverage**, and unmatched items
on both sides.

**CLI:**

```bash
python -m reqgraph compare samples/drone/drone_model.sysml samples/drone/drone_requirements.reqif
python -m reqgraph compare model.sysml reqs.csv --threshold 0.5 --report report.json --graphml matches.graphml
```

### 3.5 Compare a SysML v1 / Cameo model and build the RVTM

**GUI — card 5.** Choose the model format tab — **XMI**, **Turtle/RDF**, or a
native **Cameo / MagicDraw `.mdzip`** upload — and add the requirements. Elements
are scored by their graph neighbourhood, not only their name. The results include
the **RVTM** (Requirements Verification & Traceability Matrix):

| Status | Meaning |
|---|---|
| **Verified** | an explicit model link (satisfy / refine / derive / allocate) names the requirement **by its exact ID** |
| **Candidate** | a semantic match only — an engineer must confirm it |
| **Gap** | nothing in the model realises the requirement |

Each row also suggests a verification method (Inspection / Analysis /
Demonstration / Test). *Verified* can only come from exact IDs — fuzzy text never
produces it — so keep requirement IDs identical in the model and the document.

**Custom profiles:** custom stereotypes (e.g. `«ECU»`, `«SafetyRequirement»`) are
detected automatically; map them to roles in the GUI or with the options below.

**CLI:**

```bash
python -m reqgraph compare-v1 model.mdzip reqs.reqif --rvtm rvtm.csv --rvtm-graphml rvtm.graphml
python -m reqgraph compare-v1 model.xmi   reqs.csv --stereotype-roles "SafetyRequirement=CONSTRAINT,ECU=SUBJECT"
python -m reqgraph compare-v1 model.ttl   reqs.csv --stereotype-map profile.json --kg model_kg.graphml
```

`profile.json` example: `{"roles": {"SafetyRequirement": "CONSTRAINT"}, "relations": {"verifies": "satisfy"}}`

### 3.6 Formalise as PlusCal / TLA+

**GUI — card 6.** Enter a requirement and an ID, then **Generate PlusCal**. You get
the module (`.tla`), its TLC configuration (`.cfg`), the properties TLC will
check, and a table tracing each requirement element to the model. **Model-check
with TLC** runs the checker (needs Java and the TLA+ tools — [§4.4](#44-tla-tools-model-checking-pluscal)).

How a requirement becomes a model:

| Requirement | Model |
|---|---|
| subject | a process that performs the action |
| *When / If / As soon as* condition | a trigger the environment can raise |
| *While / During / Where* condition | a fixed (either true or false) state |
| *shall* | TLC checks the action eventually happens after the trigger (`Response`) |
| *shall … within N seconds* | also that it happens within the deadline (`DeadlineMet`) |
| *shall not* | TLC checks the action never happens (in that condition) |
| *should* | checked, but reported as advisory |
| *may* | nothing to check — a permission is not an obligation |

Not formalised, and listed under *notes* instead: rates, accuracies and distances
(*at 10 Hz*, *within 2 m*). A statement without *shall / should / may* is refused,
because what it obliges is undefined.

The generated model is a **reference model**: TLC proves the requirement is
consistent as written. To check a design, replace the generated process with your
design, keep the property names, and run TLC again.

**CLI:**

```bash
python -m reqgraph pluscal "When the door opens, the lamp shall switch on within 2 seconds." --id REQ-7
python -m reqgraph pluscal "When the door opens, the lamp shall switch on within 2 seconds." --id REQ-7 --validate
python -m reqgraph pluscal spec.docx --out specs/ --validate     # one .tla + .cfg per requirement
```

Exit code is non-zero if any requirement could not be formalised or failed TLC.

### 3.7 AI assistant: rewrite suggestions and PlusCal refinement

Turn it on first ([§4.5](#45-ai-assistant--local-ollama) or
[§4.6](#46-ai-assistant--google-cloud-vertex-ai)). Then:

- **Card 1 → ✨ Suggest rewrite** — the AI proposes an INCOSE / EARS-conformant
  rewrite. RAVEN **re-scores it with its own checks** and shows a verdict:
  *improved*, *needs your input* (the AI left a `<value>` for you to fill in —
  it is told never to invent numbers), *no change* or *regressed*. **Use this**
  copies it into the editor.
- **Card 6 → ✨ Refine with AI** — the AI improves the PlusCal model (for example,
  a more realistic environment). The candidate is **rejected** if it removes a
  property or changes the definition of one, and it is re-checked with TLC.
  **Adopt this model** appears only when it passes.

The AI never changes anything by itself. RAVEN can check scores and properties,
but not whether a rewrite kept the original meaning — review before adopting.

**CLI:**

```bash
python -m reqgraph rewrite "The system shall respond quickly." --provider ollama
python -m reqgraph rewrite reqs.csv --provider vertex --project my-project --location europe-west9
python -m reqgraph pluscal reqs.csv --out specs/ --refine --provider ollama --validate
```

---

## 4. Configure each component

### 4.1 Extraction backends

The backend decides how each requirement is split into roles. Pick it with
**Backend** in card 1 (applies to every card) or `--backend` on the CLI.

| Backend | Setup | Best for |
|---|---|---|
| `rules` (default) | none | boilerplate IREB / EARS requirements; repeatable, explainable output |
| `spacy` | `pip install -e ".[nlp]"` then `python -m spacy download en_core_web_sm` | complex or free-prose sentences |
| `bert` | `pip install -e ".[ml]"` — a trained tagger ships in `models/req_tagger` | domain vocabulary, once trained on your labelled data |

Train your own BERT tagger:

```bash
python -m reqgraph train --out models/my_tagger                         # built-in seed corpus
python -m reqgraph train --model bert-base-uncased --epochs 80 --data labelled.jsonl --out models/my_tagger
python -m reqgraph gui --model models/my_tagger                         # use it in the GUI
```

Write to a new folder: `models/req_tagger` is the shipped tagger, which the GUI
and the offline embedding fallback ([§4.3](#43-bert-embedding-similarity)) rely on.
Training downloads the base model from Hugging Face once.

`labelled.jsonl` holds one JSON object per line:
`{"text": "The pump shall start.", "spans": [[0, 8, "SUBJECT"], [9, 14, "MODALITY"], [15, 20, "PROCESS"]]}`

Whatever the backend, the text is never altered — the graph always regenerates
it exactly.

### 4.2 Templates

| Template | Structure |
|---|---|
| `IREB-Rupp` (default) | `<condition>, the <subject> <shall> [provide <actor> with the ability to] <process> <object> <constraint>` |
| `EARS` | `When/While/If/Where <trigger>, the <subject> shall <process> <object>` |

Choose with **Template** in card 1 or `--template EARS`. Custom templates (your
own modal verbs or markers) are defined in Python — see
[Custom requirement structure](reqgraph/README.md).

### 4.3 BERT embedding similarity

Used by *Connections*, *Import* and both *Compare* cards when **Similarity** is
set to **BERT embedding**, or with `--similarity embedding`.

1. Install: `pip install -e ".[ml]"`. Until then the option is shown as
   *unavailable*.
2. On first use RAVEN downloads the small `prajjwal1/bert-tiny` model from
   Hugging Face.
3. **Offline / firewalled machines:** if that download is not possible, RAVEN
   automatically uses the shipped `models/req_tagger` encoder instead (when
   started from the project folder). To use another local model, pass its folder:
   `--embedding-model /path/to/model`.

### 4.4 TLA+ tools (model-checking PlusCal)

Generating PlusCal needs nothing. **Model-checking** it needs:

1. **Java 11 or newer** on `PATH` — check with `java -version`.
2. **`tla2tools.jar`** — download it from the
   [TLA+ releases page](https://github.com/tlaplus/tlaplus/releases).
3. Tell RAVEN where it is — either set the variable:

   ```powershell
   # Windows PowerShell (current session; use System Properties for permanent)
   $env:RAVEN_TLA2TOOLS = "C:\tools\tla2tools.jar"
   ```
   ```bash
   # macOS / Linux
   export RAVEN_TLA2TOOLS=~/tools/tla2tools.jar
   ```

   …or place the jar in one of the folders RAVEN searches automatically:
   `./tla2tools.jar`, `./tools/tla2tools.jar`, `~/tla2tools.jar`,
   `~/.tlaplus/tla2tools.jar`, `/usr/share/java/tla2tools.jar`,
   `/opt/tla/tla2tools.jar`. On the CLI, `--tla2tools PATH` overrides all of these.
4. Restart the GUI. **Model-check with TLC** becomes clickable;
   `python -m reqgraph llm-status` shows `tla_tools ready`.

### 4.5 AI assistant — local Ollama

Nothing leaves your machine.

1. Install Ollama from <https://ollama.com/download>.
2. Start it and download a model:

   ```bash
   ollama serve              # the desktop app starts this for you
   ollama pull llama3.1      # or a smaller model, e.g. llama3.2, for slower machines
   ```

3. In the GUI, open **⚙ AI assistant**, choose **Ollama — local**, and click
   **Test connection**. It should report *ready* and list installed models.

| Setting | GUI field | Variable | Default |
|---|---|---|---|
| Server address | Ollama host | `OLLAMA_HOST` | `http://localhost:11434` |
| Model | Model | `RAVEN_OLLAMA_MODEL` | `llama3.1` |

**Ollama on another machine** (e.g. a GPU server): start it there with
`OLLAMA_HOST=0.0.0.0 ollama serve` so it listens on the network, then set the
host in RAVEN to `http://gpu-server:11434`. A corporate proxy is bypassed
automatically for a local Ollama.

### 4.6 AI assistant — Google Cloud Vertex AI

Requirement text is sent to Google Cloud — use this only for data you are cleared
to send.

1. **In the Google Cloud project:** enable the *Vertex AI API*
   (`gcloud services enable aiplatform.googleapis.com`) and give your account the
   **Vertex AI User** role (`roles/aiplatform.user`).
2. **Sign in on the machine that runs RAVEN** — one of:

   | Method | How |
   |---|---|
   | gcloud (simplest) | install the [gcloud CLI](https://cloud.google.com/sdk/docs/install), then `gcloud auth application-default login` |
   | Service account | `pip install -e ".[gcp]"` and set `GOOGLE_APPLICATION_CREDENTIALS` to the key file |
   | Access token | set `GOOGLE_OAUTH_ACCESS_TOKEN` (expires after about an hour) |

   RAVEN tries them in this order: access token → service account / application
   default credentials → gcloud. **Credentials are never entered in the GUI** and
   never sent to the browser.
3. In the GUI, open **⚙ AI assistant**, choose **Google Cloud — Vertex AI**, enter
   the project, and click **Test connection**. (The test makes no billable call.)

| Setting | GUI field | Variable | Default |
|---|---|---|---|
| Project ID | GCP project | `GOOGLE_CLOUD_PROJECT` | — (required) |
| Region | Location | `GOOGLE_CLOUD_LOCATION` | `us-central1` (`global` is also accepted) |
| Model | Model | `RAVEN_VERTEX_MODEL` | `gemini-2.5-flash` |
| Custom endpoint | Advanced → Endpoint override | `RAVEN_VERTEX_ENDPOINT` | Google's regional endpoint |

**Custom endpoint** — for Private Service Connect, a sovereign-cloud host or an
internal gateway, set the base URL (e.g. `https://vertex.mycompany.internal`). It
must use `https://`.

### 4.7 Make the AI settings permanent

The GUI remembers AI settings per browser. To set defaults for the GUI **and**
the CLI, use environment variables:

```powershell
# Windows PowerShell — Ollama by default
$env:RAVEN_LLM_PROVIDER = "ollama"
$env:RAVEN_OLLAMA_MODEL = "llama3.1"
python -m reqgraph gui
```

```bash
# macOS / Linux — Vertex AI by default
export RAVEN_LLM_PROVIDER=vertex
export GOOGLE_CLOUD_PROJECT=my-project
export GOOGLE_CLOUD_LOCATION=europe-west9
python -m reqgraph gui
```

On the CLI, `--provider`, `--llm-model`, `--host`, `--project`, `--location` and
`--endpoint` override the variables for one command.

---

## 5. Environment variable reference

All are optional.

| Variable | Used by | Meaning | Default |
|---|---|---|---|
| `RAVEN_LLM_PROVIDER` | AI assistant | `ollama`, `vertex`, or `none` | `none` |
| `OLLAMA_HOST` | Ollama | server address | `http://localhost:11434` |
| `RAVEN_OLLAMA_MODEL` | Ollama | model name | `llama3.1` |
| `GOOGLE_CLOUD_PROJECT` | Vertex AI | project ID (`GCLOUD_PROJECT`, `CLOUDSDK_CORE_PROJECT` also read) | — |
| `GOOGLE_CLOUD_LOCATION` | Vertex AI | region or `global` (`GOOGLE_CLOUD_REGION` also read) | `us-central1` |
| `RAVEN_VERTEX_MODEL` | Vertex AI | model name | `gemini-2.5-flash` |
| `RAVEN_VERTEX_ENDPOINT` | Vertex AI | base-URL override (https) | Google regional endpoint |
| `GOOGLE_OAUTH_ACCESS_TOKEN` | Vertex AI | explicit bearer token | — |
| `GOOGLE_APPLICATION_CREDENTIALS` | Vertex AI | service-account key file (needs the `gcp` extra) | — |
| `RAVEN_TLA2TOOLS` | PlusCal | path to `tla2tools.jar` | searched in common folders |

Setting a variable: PowerShell `$env:NAME = "value"`; Command Prompt
`set NAME=value`; macOS / Linux `export NAME=value`.

---

## 6. Input file formats

Column and attribute names are matched **case-insensitively**, with spaces and
hyphens treated as underscores — so *Object Text*, `object-text` and
`object_text` are the same.

**Requirement text** is read from the first of these columns found:
`text`, `requirement`, `requirement_text`, `req_text`, `reqif.text`,
`object_text`, `description`, `statement`, `requirement_description`,
`requirement_statement`, `body`, `content`, `shall_statement`, `primary_text`,
`specification`.

**Requirement ID** is read from the first of these found (optional, but needed for
traceability): `id`, `req_id`, `requirement_id`, `identifier`,
`reqif.foreignid`, `object_identifier`, `object_id`, `absolute_number`, `key`,
`tag`, `reference`, `req_no`, `requirement_no`, `number`, `article`.

**Every other column is kept** (rationale, priority, applicability…) and appears in
the results and downloads.

**Trace links** — these columns are read by the traceability check; a cell may
hold several IDs separated by `,` `;` `/` or `|`:

| Link | Column names |
|---|---|
| parent / derived from | `parent`, `parent_id`, `derived_from`, `derives_from`, `refines`, `trace_to`, `traces_to`, `upstream`, `source_requirement` |
| satisfies | `satisfies`, `satisfied_by`, `allocated_to`, `allocation`, `implements` |
| verification | `verified_by`, `verification`, `verification_method`, `test_case`, `verifies` |

### Per format

| Format | What RAVEN expects |
|---|---|
| **Word `.docx`** | Either a **table** whose header row has a text column (above); other tables such as revision history are skipped. Or **paragraphs starting with an ID**: `REQ-001: The system shall…`, `[SYS-12] …`, `3.1.2 The system shall…`. If neither is found, every paragraph containing *shall / must / should* is taken. Legacy `.doc` must be re-saved as `.docx`. |
| **ReqIF** `.reqif` / `.xml` | Standard ReqIF 1.x, including DOORS / Polarion exports. Text may be a STRING or **XHTML** attribute (`ReqIF.Text`, *Object Text*); ID from `ReqIF.ForeignID` / *Object Identifier*. Enumeration, date, integer, real and boolean attributes are kept. |
| **JSON** | An array — `[{"id": "R1", "text": "…", "rationale": "…"}]` — or an object keyed by ID — `{"R1": "…"}` or `{"R1": {"text": "…"}}`. |
| **CSV / Excel** | A header row with a text column (above). |
| **Plain text** | One requirement per line (paste into a card, or a `.txt` file on the CLI). |

A minimal traceable CSV:

```csv
Req ID,Requirement,Parent,Verification
SYS-1,The drone shall complete an inspection mission.,,Demonstration
SYS-2,The drone shall maintain altitude within 2 m.,SYS-1,Test
```

Sample data to try: `samples/drone/` holds a 40-requirement ReqIF set, a SysML v2
model and its knowledge graph; `sample_reqs.csv` is a small CSV set.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `python` or `pip` not found | Python not installed or not on PATH | Install Python 3.10+; on Windows tick *Add Python to PATH* |
| GUI does not start: address in use | port 8765 taken | `python -m reqgraph gui --port 9000` |
| "*could not read this as CSV … No module named 'pandas'*" | `io` extra missing | `pip install -e ".[io]"` |
| "*no requirement text column found*" | no recognised text column | rename the column to `text` or `Requirement`, or see [§6](#6-input-file-formats) |
| "*a Word document must be uploaded*" | `.docx` pasted as text | use **Upload file** |
| "*not a readable .docx file*" | legacy `.doc` or damaged file | re-save as `.docx` in Word |
| *BERT embedding — unavailable* | `ml` extra missing | `pip install -e ".[ml]"` and restart |
| "*the embedding model could not be loaded*" | no access to huggingface.co | start RAVEN from the project folder (uses `models/req_tagger`), or `--embedding-model /local/model` |
| *spacy (unavailable)* | spaCy or its model missing | `pip install -e ".[nlp]"` and `python -m spacy download en_core_web_sm` |
| **Model-check with TLC** greyed out | Java or `tla2tools.jar` not found | [§4.4](#44-tla-tools-model-checking-pluscal) |
| "*cannot formalise: … no modality*" | no *shall / should / may* | add the obligation; the completeness check flags this too |
| TLC slow for a long deadline | very fine time unit (e.g. 50 000 ms) | state the deadline in a coarser unit (50 s) |
| Ollama: "*cannot reach …11434*" | server not running | `ollama serve` (or start the Ollama app) |
| Ollama: "*model … is not pulled*" | model not downloaded | `ollama pull <model>` |
| Ollama answers very slowly | large model on CPU | use a smaller model (`llama3.2`) |
| Vertex: "*no Google Cloud credentials*" | not signed in | `gcloud auth application-default login` ([§4.6](#46-ai-assistant--google-cloud-vertex-ai)) |
| Vertex: "*no GCP project*" | project not set | set it in the AI assistant card or `GOOGLE_CLOUD_PROJECT` |
| Vertex: HTTP 403 | API disabled or missing role | enable `aiplatform.googleapis.com`; grant *Vertex AI User* |
| Vertex: HTTP 404 on the model | model not available in that region | try `GOOGLE_CLOUD_LOCATION=global` or another model |
| "*requests must be sent as application/json*" (HTTP 403) | calling the GUI API from a script or another site | from scripts, send `Content-Type: application/json` to `http://127.0.0.1:<port>`; cross-site calls are blocked by design |

Still stuck? `python -m reqgraph llm-status` summarises the AI and TLA+ setup, and
`python -m reqgraph -v <command>` prints more detail.

---

## 8. Tests and further documentation

Run the test suite:

```bash
pip install -e ".[dev,io]"
python -m pytest tests/ -q
```

Tests for optional components (spaCy, BERT, TLC) are skipped when those
components are not installed. Set `RAVEN_TLA2TOOLS` to include the TLC
model-checking tests.

| Document | Contents |
|---|---|
| [reqgraph/README.md](reqgraph/README.md) | developer reference: Python API, backends in depth, exporters |
| [reqgraph/docs/MANUAL.md](reqgraph/docs/MANUAL.md) | function-by-function reference and limitations |
| [reqgraph/docs/ARCHITECTURE.md](reqgraph/docs/ARCHITECTURE.md) | architecture and data flow |
| GUI → **Constraints & roadmap** | per-function constraints, workarounds and plans |
| [docs/RAVEN_CTO_deck.html](docs/RAVEN_CTO_deck.html) | 7-slide overview deck (open in a browser) |
