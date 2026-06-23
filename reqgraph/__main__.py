"""
reqgraph command-line interface.

    python -m reqgraph parse  "<requirement text>" [options]
    python -m reqgraph batch  <in.csv|in.xlsx|in.reqif> [--out out.csv|.xlsx|.reqif|.json]
    python -m reqgraph train  [--model NAME] [--epochs N] [--out DIR]
    python -m reqgraph analyze <in.csv|in.xlsx|in.reqif>

Run any subcommand with -h for its options.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from . import __version__
from .core import Role
from .errors import ReqGraphError
from .extractors import BertTaggerExtractor, RuleExtractor, SpacyExtractor
from .parser import RequirementParser
from .quality import enrich
from .templates import RUPP_TEMPLATE, TEMPLATES


# --- helpers ---------------------------------------------------------------

def _build_parser(template_name, backend, model):
    template = TEMPLATES.get(template_name, RUPP_TEMPLATE)
    if backend == "spacy":
        ex = SpacyExtractor()
    elif backend == "bert":
        if not model:
            sys.exit("error: --backend bert requires --model <saved-tagger-dir>")
        ex = BertTaggerExtractor(model_dir=model)
    else:
        ex = RuleExtractor()
    return RequirementParser(template, ex)


def _read_items(path):
    from .io_formats import (read_reqif, read_requirements_csv,
                             read_requirements_excel, read_requirements_json)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return read_requirements_csv(path)
    if ext in (".xlsx", ".xls"):
        return read_requirements_excel(path)
    if ext in (".reqif", ".xml"):
        return read_reqif(path)
    if ext == ".json":
        return read_requirements_json(path)
    sys.exit(f"error: unsupported input extension {ext!r}")


# --- subcommands -----------------------------------------------------------

def cmd_parse(args):
    p = _build_parser(args.template, args.backend, args.model)
    segments = [args.text] if args.no_split else p.split(args.text)
    multi = len(segments) > 1
    if multi:
        print(f"detected {len(segments)} requirements in the input "
              f"(use --no-split to parse as one)\n")
    fmt = args.format
    if fmt == "json" and multi:
        graphs = p.parse_many(args.text, metadata={"id": args.id} if args.id else None)
        if args.analyze:
            for g in graphs:
                enrich(g)
        print(json.dumps([g.to_dict() for g in graphs], indent=2))
        return 0
    for i, seg in enumerate(segments, 1):
        rid = f"{args.id}-{i}" if (args.id and multi) else args.id
        g = p.parse(seg, metadata={"id": rid} if rid else None)
        if args.analyze:
            enrich(g)
        if multi:
            print(f"--- requirement {i}/{len(segments)}: {seg}")
        if fmt == "summary":
            print(g.summary())
            print("round-trip exact:", g.generate() == seg)
        elif fmt == "json":
            print(g.to_json())
        elif fmt == "mermaid":
            print(g.to_mermaid())
        elif fmt == "dot":
            print(g.to_dot())
        elif fmt == "cypher":
            print(g.to_cypher())
        elif fmt == "graphml":
            print(g.to_graphml())
        elif fmt == "turtle":
            print(g.to_turtle())
        elif fmt == "elements":
            for n in g.elements():
                print(f"{n.role.value:<11}\t{n.text.strip()}")
        if multi and i < len(segments):
            print()
    return 0


def cmd_batch(args):
    from .io_formats import (requirements_to_dataframe, write_csv, write_excel,
                             write_reqif)
    items = _read_items(args.infile)
    ex = None
    if args.backend == "spacy":
        ex = SpacyExtractor()
    elif args.backend == "bert":
        ex = BertTaggerExtractor(model_dir=args.model)
    template = TEMPLATES.get(args.template, RUPP_TEMPLATE)

    out = args.out
    if not out:
        df = requirements_to_dataframe(items, template=template, extractor=ex)
        print(df.to_string(index=False))
        return 0
    ext = os.path.splitext(out)[1].lower()
    if ext == ".reqif":
        write_reqif(items, out)
    elif ext in (".xlsx", ".xls"):
        write_excel(items, out, template=template, extractor=ex)
    elif ext == ".json":
        df = requirements_to_dataframe(items, template=template, extractor=ex)
        df.to_json(out, orient="records", indent=2)
    else:
        write_csv(items, out, template=template, extractor=ex)
    print(f"wrote {len(items)} requirements -> {out}")
    return 0


def cmd_train(args):
    from .nlp import BertTokenTagger
    if args.data:
        examples = _load_jsonl(args.data)
    else:
        from .seed_data import training_examples
        examples = training_examples()
    print(f"training {args.model} on {len(examples)} labelled requirements "
          f"({args.epochs} epochs)...")
    tagger = BertTokenTagger(model_name=args.model).train(
        examples, epochs=args.epochs, lr=args.lr)
    tagger.save(args.out)
    print(f"saved tagger -> {args.out}")
    print("try:  python -m reqgraph parse \"...\" --backend bert --model", args.out)
    return 0


def _load_jsonl(path):
    """Each line: {\"text\": ..., \"spans\": [[start, end, \"ROLE\"], ...]}."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            spans = [(s, e, Role(r)) for s, e, r in rec["spans"]]
            out.append((rec["text"], spans))
    return out


def cmd_gui(args):
    from .gui import launch
    return launch(port=args.port, open_browser=not args.no_browser,
                  bert_model_dir=args.model)


def cmd_connections(args):
    from .corpus import build_requirement_set_graph
    items = _read_items(args.infile)
    ex = None
    if args.backend == "spacy":
        ex = SpacyExtractor()
    elif args.backend == "bert":
        ex = BertTaggerExtractor(model_dir=args.model)
    template = TEMPLATES.get(args.template, RUPP_TEMPLATE)
    roles = tuple(Role(r.strip().upper()) for r in args.roles.split(","))

    rsg = build_requirement_set_graph(
        items, template=template, extractor=ex, roles=roles,
        threshold=args.threshold, similarity=args.similarity,
        embedding_model=args.embedding_model)

    fmt = args.format
    if fmt == "json":
        print(json.dumps(rsg.to_dict(), indent=2))
    elif fmt == "mermaid":
        print(rsg.to_mermaid())
    elif fmt == "dot":
        print(rsg.to_dot())
    elif fmt == "cypher":
        print(rsg.to_cypher())
    elif fmt == "graphml":
        print(rsg.to_graphml())
    elif fmt == "turtle":
        print(rsg.to_turtle())
    else:
        print(f"loaded {len(rsg.req_ids)} requirements from {args.infile}\n")
        if not rsg.connections:
            print(f"no cross-requirement connections found "
                  f"(similarity={args.similarity}, threshold={args.threshold})")
        for c in rsg._dedup_pairs():
            print(f"[{c.a.req_id}] {c.role.value} {c.a.text!r}  "
                  f"~{c.score:.2f}~  [{c.b.req_id}] {c.role.value} {c.b.text!r}")
    return 0


def cmd_analyze(args):
    items = _read_items(args.infile)
    texts = [t for _id, t, *_ in items]
    print(f"loaded {len(texts)} requirements from {args.infile}\n")

    # per-requirement quality / type / EARS
    p = RequirementParser(RUPP_TEMPLATE)
    print("== quality / type / EARS ==")
    for (rid, text, *_) in items:
        g = enrich(p.parse(text))
        q = g.analysis["quality"]
        smells = [k for k, v in q.items() if v and k != "weak_words"]
        if q["weak_words"]:
            smells.append("weak_words=" + ",".join(q["weak_words"]))
        print(f"[{rid or '?'}] type={g.analysis['type']}, "
              f"ears={g.analysis['ears_pattern']}, smells={smells or 'none'}")

    # embedding-based duplicates / conflicts (optional)
    try:
        from .nlp import RequirementAnalyzer
        az = RequirementAnalyzer(model_name=args.model)
        print("\n== near-duplicates (cosine >= %.2f) ==" % args.dup_threshold)
        for i, j, s in az.find_duplicates(texts, threshold=args.dup_threshold):
            print(f"  [{i}] ~ [{j}]  {s:.3f}")
        print("== potential conflicts (similar + opposite polarity) ==")
        for i, j, s in az.detect_conflicts(texts, threshold=args.conflict_threshold):
            print(f"  [{i}] <> [{j}]  {s:.3f}")
    except Exception as e:
        print("\n(embedding analysis skipped:", e, ")")
    return 0


def cmd_export(args):
    """Parse + quality analysis + connections → CSV, JSON, consolidated GraphML."""
    from .corpus import build_requirement_set_graph

    items = _read_items(args.infile)
    if not items:
        sys.exit("error: no requirements found in input file")

    ex = None
    if args.backend == "spacy":
        ex = SpacyExtractor()
    elif args.backend == "bert":
        if not args.model:
            sys.exit("error: --backend bert requires --model")
        ex = BertTaggerExtractor(model_dir=args.model)
    template = TEMPLATES.get(args.template, RUPP_TEMPLATE)
    roles = tuple(Role(r.strip().upper()) for r in args.roles.split(","))

    prefix = args.out_prefix
    csv_path   = args.csv     or (f"{prefix}.csv"     if prefix else None)
    json_path  = args.json    or (f"{prefix}.json"    if prefix else None)
    gml_path   = args.graphml or (f"{prefix}.graphml" if prefix else None)

    if not any([csv_path, json_path, gml_path]):
        sys.exit("error: specify --out-prefix or at least one of --csv / --json / --graphml")

    # build set graph: parses every requirement and finds cross-req connections
    rsg = build_requirement_set_graph(
        items, template=template, extractor=ex, roles=roles,
        threshold=args.threshold, similarity=args.similarity,
        embedding_model=args.embedding_model)

    # flat quality + metadata table, derived from the already-parsed (and
    # compound-split) graphs so the CSV/JSON row set matches the GraphML.
    # Built lazily so a GraphML-only export needs no pandas.
    df = rsg.to_dataframe() if (csv_path or json_path) else None

    written = []
    if csv_path:
        df.to_csv(csv_path, index=False)
        written.append(csv_path)
    if json_path:
        df.to_json(json_path, orient="records", indent=2, force_ascii=False)
        written.append(json_path)
    if gml_path:
        with open(gml_path, "w", encoding="utf-8") as fh:
            fh.write(rsg.to_element_graphml())
        written.append(gml_path)

    print(f"exported {len(rsg.req_ids)} requirements "
          f"({len(rsg.connections)} connections found)")
    for p in written:
        print(f"  -> {p}")
    return 0


# --- argument parser -------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="reqgraph",
                                 description="IREB-CPRE requirement <-> graph toolkit")
    ap.add_argument("--version", action="version", version=f"reqgraph {__version__}")
    ap.add_argument("-v", "--verbose", action="count", default=0,
                    help="-v info, -vv debug (default: warnings only)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("parse", help="parse one requirement into a graph")
    pp.add_argument("text")
    pp.add_argument("--template", default="IREB-Rupp", help="template name (default IREB-Rupp)")
    pp.add_argument("--backend", default="rules", choices=["rules", "spacy", "bert"])
    pp.add_argument("--model", help="saved tagger dir (for --backend bert)")
    pp.add_argument("--format", default="summary",
                    choices=["summary", "json", "mermaid", "dot", "cypher",
                             "graphml", "turtle", "elements"])
    pp.add_argument("--analyze", action="store_true", help="add quality/type/EARS")
    pp.add_argument("--id", help="requirement id metadata")
    pp.add_argument("--no-split", action="store_true",
                    help="treat the input as one requirement even if it "
                         "contains several modality clauses")
    pp.set_defaults(func=cmd_parse)

    pb = sub.add_parser("batch", help="parse a CSV/Excel/ReqIF set")
    pb.add_argument("infile")
    pb.add_argument("--out", help="output .csv/.xlsx/.reqif/.json (default: print table)")
    pb.add_argument("--template", default="IREB-Rupp")
    pb.add_argument("--backend", default="rules", choices=["rules", "spacy", "bert"])
    pb.add_argument("--model")
    pb.set_defaults(func=cmd_batch)

    pt = sub.add_parser("train", help="fine-tune and save a BERT element tagger")
    pt.add_argument("--model", default="prajjwal1/bert-tiny",
                    help="base checkpoint (use bert-base-uncased for production)")
    pt.add_argument("--epochs", type=int, default=60)
    pt.add_argument("--lr", type=float, default=5e-4)
    pt.add_argument("--out", default="models/req_tagger", help="output dir")
    pt.add_argument("--data", help="JSONL of {text, spans:[[s,e,ROLE]...]}; "
                                   "defaults to the built-in seed corpus")
    pt.set_defaults(func=cmd_train)

    pg = sub.add_parser("gui", help="launch the local web GUI (zero dependencies)")
    pg.add_argument("--port", type=int, default=8765)
    pg.add_argument("--model", default=None,
                    help="saved BERT tagger dir (default: models/req_tagger)")
    pg.add_argument("--no-browser", action="store_true",
                    help="do not auto-open the browser")
    pg.set_defaults(func=cmd_gui)

    pa = sub.add_parser("analyze", help="quality + duplicate/conflict report for a set")
    pa.add_argument("infile")
    pa.add_argument("--model", default="prajjwal1/bert-tiny", help="embedding model")
    pa.add_argument("--dup-threshold", type=float, default=0.95)
    pa.add_argument("--conflict-threshold", type=float, default=0.9)
    pa.set_defaults(func=cmd_analyze)

    px = sub.add_parser(
        "export",
        help="parse + quality + connections → CSV, JSON, consolidated element GraphML")
    px.add_argument("infile",
                    help="input file (.csv, .xlsx, .xls, .json, .reqif, .xml)")
    px.add_argument("--out-prefix", default=None,
                    help="output path prefix; generates <prefix>.csv, <prefix>.json, "
                         "<prefix>.graphml (overridden by individual flags)")
    px.add_argument("--csv",     metavar="PATH", default=None,
                    help="explicit CSV output path")
    px.add_argument("--json",    metavar="PATH", default=None,
                    help="explicit JSON output path")
    px.add_argument("--graphml", metavar="PATH", default=None,
                    help="explicit consolidated element GraphML output path")
    px.add_argument("--template",  default="IREB-Rupp")
    px.add_argument("--backend",   default="rules", choices=["rules", "spacy", "bert"])
    px.add_argument("--model",     default=None,
                    help="saved tagger dir (for --backend bert)")
    px.add_argument("--roles",     default="SUBJECT,OBJECT",
                    help="roles for cross-req similarity (default SUBJECT,OBJECT)")
    px.add_argument("--similarity", default="lexical", choices=["lexical", "embedding"])
    px.add_argument("--threshold",  type=float, default=0.6)
    px.add_argument("--embedding-model", default="prajjwal1/bert-tiny")
    px.set_defaults(func=cmd_export)

    pc = sub.add_parser("connections",
                        help="find similar SUBJECT/OBJECT elements across a "
                             "requirement set and visualize the connections")
    pc.add_argument("infile")
    pc.add_argument("--template", default="IREB-Rupp")
    pc.add_argument("--backend", default="rules", choices=["rules", "spacy", "bert"])
    pc.add_argument("--model")
    pc.add_argument("--roles", default="SUBJECT,OBJECT",
                    help="comma-separated roles to cross-reference (default SUBJECT,OBJECT)")
    pc.add_argument("--similarity", default="lexical", choices=["lexical", "embedding"])
    pc.add_argument("--threshold", type=float, default=0.6)
    pc.add_argument("--embedding-model", default="prajjwal1/bert-tiny")
    pc.add_argument("--format", default="text",
                    choices=["text", "json", "mermaid", "dot", "cypher", "graphml", "turtle"])
    pc.set_defaults(func=cmd_connections)

    args = ap.parse_args(argv)
    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except ReqGraphError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
