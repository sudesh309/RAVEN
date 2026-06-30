"""
reqgraph command-line interface.

    python -m reqgraph parse    "<requirement text>" [options]
    python -m reqgraph batch    <in.csv|in.xlsx|in.reqif> [--out out.csv|.xlsx|.reqif|.json]
    python -m reqgraph train    [--model NAME] [--epochs N] [--out DIR]
    python -m reqgraph analyze  <in.csv|in.xlsx|in.reqif>
    python -m reqgraph compare  <model.sysml> <requirements.csv|.json|.txt>

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
    from .errors import DataFormatError
    from .io_formats import (read_reqif, read_requirements_csv,
                             read_requirements_excel, read_requirements_json)
    if not os.path.isfile(path):
        sys.exit(f"error: input file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    def _read_plain_text(path):
        with open(path, encoding="utf-8") as fh:
            return [ln.strip() for ln in fh if ln.strip()]

    readers = {".csv": read_requirements_csv, ".json": read_requirements_json,
               ".xlsx": read_requirements_excel, ".xls": read_requirements_excel,
               ".reqif": read_reqif, ".xml": read_reqif,
               ".txt": _read_plain_text}
    if ext not in readers:
        sys.exit(f"error: unsupported input extension {ext!r}; use one of "
                 f".csv .xlsx .xls .json .reqif .xml .txt")
    try:
        return readers[ext](path)
    except DataFormatError as exc:
        sys.exit(f"error: {exc}")
    except Exception as exc:                       # malformed file, bad encoding…
        sys.exit(f"error: could not read {path}: {exc}")


def _parse_roles(spec):
    """Parse a --roles string like 'SUBJECT,OBJECT' into Role enums."""
    try:
        return tuple(Role(r.strip().upper()) for r in spec.split(",") if r.strip())
    except ValueError as exc:
        sys.exit(f"error: unknown role {exc}; choose from "
                 f"{', '.join(r.value for r in Role)}")


def _validate_threshold(value):
    if value < 0 or value > 1:
        clamped = min(1.0, max(0.0, value))
        print(f"warning: threshold {value:g} is outside 0–1; clamped to {clamped:g}",
              file=sys.stderr)
        return clamped
    return value


def _read_sysml_model(path: str):
    """Read a SysML v2 ``.sysml`` / ``.kerml`` file with a friendly error."""
    from .errors import DataFormatError
    from .sysml_parser import read_sysml
    if not os.path.isfile(path):
        sys.exit(f"error: SysML model file not found: {path!r}")
    try:
        return read_sysml(path)
    except DataFormatError as exc:
        sys.exit(f"error: {exc}")
    except Exception as exc:
        sys.exit(f"error: could not read {path}: {exc}")


def _read_sysml_v1_model(path: str, stereotype_roles=None,
                         stereotype_relations=None):
    """Read a SysML v1 XMI, Turtle, or Cameo .mdzip file with a friendly error."""
    from .errors import DataFormatError
    from .sysml_v1_parser import read_sysml_v1
    if not os.path.isfile(path):
        sys.exit(f"error: SysML v1 model file not found: {path!r}")
    try:
        return read_sysml_v1(path, stereotype_roles=stereotype_roles,
                             stereotype_relations=stereotype_relations)
    except DataFormatError as exc:
        sys.exit(f"error: {exc}")
    except ImportError as exc:
        sys.exit(f"error: {exc}")
    except Exception as exc:
        sys.exit(f"error: could not read {path}: {exc}")


def _load_stereotype_map(args):
    """Build (roles, relations) override dicts from --stereotype-map / -roles."""
    roles, rels = {}, {}
    path = getattr(args, "stereotype_map", None)
    if path:
        import json
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            sys.exit(f"error: could not read --stereotype-map {path!r}: {exc}")
        if isinstance(data, dict) and ("roles" in data or "relations" in data):
            roles.update(data.get("roles") or {})
            rels.update(data.get("relations") or {})
        elif isinstance(data, dict):
            _ROLE_NAMES = {"SUBJECT", "OBJECT", "PROCESS", "CONDITION", "ACTOR",
                           "CONSTRAINT", "MODALITY", "OPERATOR", "DETAILS"}
            for k, v in data.items():
                (roles if str(v).strip().upper() in _ROLE_NAMES else rels)[k] = v
    spec = getattr(args, "stereotype_roles", None)
    if spec:
        for pair in spec.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                roles[k.strip()] = v.strip()
    return (roles or None), (rels or None)


def _build_set_graph_cli(items, *, template, extractor, roles, threshold,
                         similarity, embedding_model):
    """build_requirement_set_graph with a friendly message if the optional
    embedding backend's dependencies are missing."""
    from .corpus import build_requirement_set_graph
    try:
        return build_requirement_set_graph(
            items, template=template, extractor=extractor, roles=roles,
            threshold=threshold, similarity=similarity,
            embedding_model=embedding_model)
    except ImportError as exc:
        sys.exit(f"error: --similarity embedding needs PyTorch + transformers "
                 f"installed ({exc}); use --similarity lexical (no extra deps)")


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
    items = _read_items(args.infile)
    if not items:
        sys.exit("error: no requirements found in input file")
    ex = None
    if args.backend == "spacy":
        ex = SpacyExtractor()
    elif args.backend == "bert":
        ex = BertTaggerExtractor(model_dir=args.model)
    template = TEMPLATES.get(args.template, RUPP_TEMPLATE)
    roles = _parse_roles(args.roles)

    rsg = _build_set_graph_cli(
        items, template=template, extractor=ex, roles=roles,
        threshold=_validate_threshold(args.threshold), similarity=args.similarity,
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
    roles = _parse_roles(args.roles)

    prefix = args.out_prefix
    csv_path   = args.csv     or (f"{prefix}.csv"     if prefix else None)
    json_path  = args.json    or (f"{prefix}.json"    if prefix else None)
    gml_path   = args.graphml or (f"{prefix}.graphml" if prefix else None)
    rttl_path  = args.req_turtle or (f"{prefix}.req.ttl" if prefix else None)

    if not any([csv_path, json_path, gml_path, rttl_path]):
        sys.exit("error: specify --out-prefix or at least one of "
                 "--csv / --json / --graphml / --req-turtle")

    # build set graph: parses every requirement and finds cross-req connections
    rsg = _build_set_graph_cli(
        items, template=template, extractor=ex, roles=roles,
        threshold=_validate_threshold(args.threshold), similarity=args.similarity,
        embedding_model=args.embedding_model)

    # flat quality + metadata table, derived from the already-parsed (and
    # compound-split) graphs so the CSV/JSON row set matches the GraphML.
    # Built lazily so a GraphML-only export needs no pandas.
    df = None
    if csv_path or json_path:
        try:
            df = rsg.to_dataframe()
        except ImportError as exc:
            sys.exit(f"error: CSV/JSON output needs pandas installed ({exc}); "
                     f"install it with `pip install reqgraph[io]`, or export only "
                     f"GraphML with --graphml")

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
    if rttl_path:
        with open(rttl_path, "w", encoding="utf-8") as fh:
            fh.write(rsg.to_req_turtle())
        written.append(rttl_path)

    print(f"exported {len(rsg.req_ids)} requirements "
          f"({len(rsg.connections)} connections found)")
    for p in written:
        print(f"  -> {p}")
    return 0


def cmd_compare(args):
    """Semantically compare a SysML v2 model against a requirement set."""
    from pathlib import Path
    model = _read_sysml_model(args.model_file)
    items = _read_items(args.req_file)
    if not items:
        sys.exit("error: no requirements found in requirement file")

    ex = None
    if args.backend == "spacy":
        ex = SpacyExtractor()
    elif args.backend == "bert":
        if not args.model:
            sys.exit("error: --backend bert requires --model")
        ex = BertTaggerExtractor(model_dir=args.model)
    template = TEMPLATES.get(args.template, RUPP_TEMPLATE)
    roles = _parse_roles(args.roles)
    threshold = _validate_threshold(args.threshold)

    from .sysml_compare import compare as _compare
    try:
        report = _compare(model, items, roles=roles, threshold=threshold,
                          similarity=args.similarity,
                          embedding_model=args.embedding_model,
                          template=template, extractor=ex)
    except ImportError as exc:
        sys.exit(f"error: --similarity embedding needs PyTorch + transformers "
                 f"installed ({exc}); use --similarity lexical")

    for w in report.warnings:
        print(f"warning: {w}", file=sys.stderr)

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        # Human-readable summary
        def pct(v):
            return f"{v:.1%}"
        print(f"\nSemantic match:    {pct(report.semantic_match)}")
        print(f"Model coverage:    {pct(report.model_coverage)}"
              f"  ({report.n_model_elements} model elements)")
        print(f"Req coverage:      {pct(report.req_coverage)}"
              f"  ({report.n_req_elements} req elements)")
        if report.role_breakdown:
            print("\nRole breakdown:")
            for role_val, stats in report.role_breakdown.items():
                print(f"  {role_val:<15}  "
                      f"model={pct(stats['model_coverage'])}  "
                      f"req={pct(stats['req_coverage'])}  "
                      f"n_model={stats['n_model']}  n_req={stats['n_req']}")
        if report.matches:
            print(f"\nMatches ({len(report.matches)}):")
            for m in report.matches[:20]:
                print(f"  [{m.sysml_element.element_type}] {m.sysml_element.name}"
                      f"  --{m.role.value} {m.score:.2f}-->  "
                      f"[{m.req_id}] {m.req_text!r}")
            if len(report.matches) > 20:
                print(f"  … {len(report.matches) - 20} more (use --format json)")
        if report.unmatched_model:
            print(f"\nModel elements NOT covered by any requirement "
                  f"({len(report.unmatched_model)}):")
            for e in report.unmatched_model:
                print(f"  [{e.element_type}] {e.name}"
                      + (f"  (in {e.package})" if e.package else ""))
        if report.unmatched_reqs:
            print(f"\nRequirement elements NOT found in model "
                  f"({len(report.unmatched_reqs)}):")
            for req_id, role_val, txt in report.unmatched_reqs:
                print(f"  {req_id}  [{role_val}]  {txt!r}")

    if args.report:
        Path(args.report).write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nreport -> {args.report}")
    if args.graphml:
        Path(args.graphml).write_text(report.to_graphml(), encoding="utf-8")
        print(f"graphml -> {args.graphml}")
    return 0


def cmd_compare_v1(args):
    """Semantically compare a SysML v1 model against a requirement set."""
    from pathlib import Path
    stereo_roles, stereo_rels = _load_stereotype_map(args)
    model = _read_sysml_v1_model(args.model_file, stereotype_roles=stereo_roles,
                                 stereotype_relations=stereo_rels)
    items = _read_items(args.req_file)
    if not items:
        sys.exit("error: no requirements found in requirement file")

    ex = None
    if args.backend == "spacy":
        ex = SpacyExtractor()
    elif args.backend == "bert":
        if not args.model:
            sys.exit("error: --backend bert requires --model")
        ex = BertTaggerExtractor(model_dir=args.model)
    template = TEMPLATES.get(args.template, RUPP_TEMPLATE)
    roles = _parse_roles(args.roles)
    threshold = _validate_threshold(args.threshold)

    from .sysml_v1_compare import compare_v1 as _compare_v1
    try:
        report = _compare_v1(model, items, roles=roles, threshold=threshold,
                             similarity=args.similarity,
                             embedding_model=args.embedding_model,
                             context_hops=args.context_hops,
                             template=template, extractor=ex)
    except ImportError as exc:
        sys.exit(f"error: --similarity embedding needs PyTorch + transformers "
                 f"installed ({exc}); use --similarity lexical")

    for w in report.warnings:
        print(f"warning: {w}", file=sys.stderr)

    # Certification traceability matrix (RVTM)
    from .traceability import build_traceability_matrix
    tm = build_traceability_matrix(model, items, report, template=template,
                                   extractor=ex, roles=roles,
                                   candidate_threshold=threshold)

    if args.format == "json":
        out = report.to_dict()
        out["traceability"] = tm.to_dict()
        print(json.dumps(out, indent=2))
    else:
        def pct(v):
            return f"{v:.1%}"
        print(f"\nSysML v1 Semantic match:  {pct(report.semantic_match)}")
        print(f"Model coverage:           {pct(report.model_coverage)}"
              f"  ({report.n_model_elements} model elements)")
        print(f"Req coverage:             {pct(report.req_coverage)}"
              f"  ({report.n_req_elements} req elements)")
        if report.role_breakdown:
            print("\nRole breakdown:")
            for role_val, stats in report.role_breakdown.items():
                print(f"  {role_val:<15}  "
                      f"model={pct(stats['model_coverage'])}  "
                      f"req={pct(stats['req_coverage'])}  "
                      f"n_model={stats['n_model']}  n_req={stats['n_req']}")
        if report.matches:
            print(f"\nTop matches ({min(20, len(report.matches))}):")
            for m in report.matches[:20]:
                print(f"  [{m.element.stereotype or m.element.element_type}] "
                      f"{m.element.name}"
                      f"  conf={m.confidence:.2f} "
                      f"(name={m.name_score:.2f}, ctx={m.context_score:.2f}"
                      + (", sat" if m.via_satisfaction else "") + ")"
                      f"  --{m.role.value}-->  [{m.req_id}] {m.req_text!r}")
            if len(report.matches) > 20:
                print(f"  … {len(report.matches) - 20} more (use --format json)")
        if report.unmatched_model:
            print(f"\nModel elements NOT covered ({len(report.unmatched_model)}):")
            for e in report.unmatched_model[:10]:
                print(f"  [{e.stereotype or e.element_type}] {e.name}"
                      + (f"  (in {e.package})" if e.package else ""))
        if report.unmatched_reqs:
            print(f"\nRequirement elements NOT in model ({len(report.unmatched_reqs)}):")
            for req_id, role_val, txt in report.unmatched_reqs[:10]:
                print(f"  {req_id}  [{role_val}]  {txt!r}")

        # Certification traceability summary
        print(f"\nCertification traceability (RVTM):")
        print(f"  Auditable trace rate:   {pct(tm.explicit_trace_rate())}"
              f"  ({tm.n_verified}/{tm.n_requirements} via explicit model links)")
        print(f"  Verified / Candidate / Gap: "
              f"{tm.n_verified} / {tm.n_candidate} / {tm.n_gap}")
        print(f"  Quality-passing:        {tm.n_quality_pass}/{tm.n_requirements}")
        print(f"  Verification readiness: {pct(tm.verification_readiness())}"
              f"  (traced AND quality-passing)")
        if tm.findings:
            high = sum(1 for f in tm.findings if f.severity == "high")
            print(f"  Findings: {len(tm.findings)} ({high} high)")
            for f in [f for f in tm.findings if f.severity == "high"][:8]:
                print(f"    [{f.finding_id}] {f.subject}: {f.issue}")

    if args.rvtm:
        Path(args.rvtm).write_text(tm.to_csv(), encoding="utf-8")
        print(f"\nRVTM (CSV) -> {args.rvtm}")
    if args.rvtm_graphml:
        Path(args.rvtm_graphml).write_text(tm.to_graphml(), encoding="utf-8")
        print(f"RVTM graphml -> {args.rvtm_graphml}")
    if args.report:
        Path(args.report).write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nreport -> {args.report}")
    if args.graphml:
        Path(args.graphml).write_text(report.to_graphml(), encoding="utf-8")
        print(f"graphml -> {args.graphml}")
    if args.kg:
        Path(args.kg).write_text(model.to_graphml(), encoding="utf-8")
        print(f"model KG graphml -> {args.kg}")
    if args.out_turtle:
        Path(args.out_turtle).write_text(model.to_turtle(), encoding="utf-8")
        print(f"model turtle -> {args.out_turtle}")
    if args.ontology_graphml and report.ontology_diff:
        Path(args.ontology_graphml).write_text(
            report.ontology_diff.to_graphml(), encoding="utf-8")
        print(f"ontology graphml -> {args.ontology_graphml}")
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
        help="parse + quality + connections → CSV, JSON, GraphML, requirements Turtle")
    px.add_argument("infile",
                    help="input file (.csv, .xlsx, .xls, .json, .reqif, .xml)")
    px.add_argument("--out-prefix", default=None,
                    help="output path prefix; generates <prefix>.csv, <prefix>.json, "
                         "<prefix>.graphml, <prefix>.req.ttl "
                         "(overridden by individual flags)")
    px.add_argument("--csv",     metavar="PATH", default=None,
                    help="explicit CSV output path")
    px.add_argument("--json",    metavar="PATH", default=None,
                    help="explicit JSON output path")
    px.add_argument("--graphml", metavar="PATH", default=None,
                    help="explicit consolidated element GraphML output path")
    px.add_argument("--req-turtle", metavar="PATH", default=None,
                    help="explicit requirements Turtle/RDF ontology output path "
                         "(reqont: ontology)")
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

    pcmp = sub.add_parser(
        "compare",
        help="semantically compare a SysML v2 model against a requirement set")
    pcmp.add_argument("model_file",
                      help="SysML v2 model (.sysml or .kerml)")
    pcmp.add_argument("req_file",
                      help="requirement file (.csv, .xlsx, .xls, .json, .reqif, .xml, "
                           "or plain .txt)")
    pcmp.add_argument("--threshold",  type=float, default=0.6,
                      help="similarity cut-off 0–1 (default 0.6)")
    pcmp.add_argument("--similarity", default="lexical", choices=["lexical", "embedding"],
                      help="'lexical' (default, no deps) or 'embedding' (needs torch)")
    pcmp.add_argument("--embedding-model", default="prajjwal1/bert-tiny",
                      help="HuggingFace model for embedding similarity")
    pcmp.add_argument("--roles", default="SUBJECT,PROCESS,OBJECT,CONDITION",
                      help="comma-separated semantic roles to compare "
                           "(default SUBJECT,PROCESS,OBJECT,CONDITION)")
    pcmp.add_argument("--format", default="text", choices=["text", "json"],
                      help="output format: human-readable text (default) or JSON")
    pcmp.add_argument("--report",  metavar="PATH", default=None,
                      help="write full JSON report to this file")
    pcmp.add_argument("--graphml", metavar="PATH", default=None,
                      help="write bipartite match GraphML to this file")
    pcmp.add_argument("--template", default="IREB-Rupp")
    pcmp.add_argument("--backend",  default="rules", choices=["rules", "spacy", "bert"])
    pcmp.add_argument("--model",    default=None,
                      help="saved tagger dir (for --backend bert)")
    pcmp.set_defaults(func=cmd_compare)

    pcv1 = sub.add_parser(
        "compare-v1",
        help="context-aware comparison of a SysML v1 XMI/Turtle model "
             "against a requirement set (uses graph neighborhood context)")
    pcv1.add_argument("model_file",
                      help="SysML v1 model (.xmi, .uml, .xml, .ttl, .rdf, .owl, "
                           ".n3, or a Cameo/MagicDraw .mdzip archive)")
    pcv1.add_argument("req_file",
                      help="requirement file (.csv, .xlsx, .xls, .json, .reqif, "
                           ".xml, or plain .txt)")
    pcv1.add_argument("--stereotype-map", metavar="PATH", default=None,
                      help="JSON file mapping custom-profile stereotypes to IREB "
                           "roles / trace verbs, e.g. "
                           '{"roles":{"SafetyRequirement":"CONSTRAINT"},'
                           '"relations":{"verifies":"satisfy"}}')
    pcv1.add_argument("--stereotype-roles", metavar="SPEC", default=None,
                      help="inline custom stereotype→role overrides, e.g. "
                           "'SafetyRequirement=CONSTRAINT,ECU=SUBJECT'")
    pcv1.add_argument("--threshold",  type=float, default=0.5,
                      help="confidence cut-off 0–1 (default 0.5)")
    pcv1.add_argument("--similarity", default="lexical",
                      choices=["lexical", "embedding"],
                      help="'lexical' (default) or 'embedding' (needs torch)")
    pcv1.add_argument("--embedding-model", default="prajjwal1/bert-tiny")
    pcv1.add_argument("--roles", default="SUBJECT,PROCESS,OBJECT,CONDITION",
                      help="comma-separated roles (default SUBJECT,PROCESS,OBJECT,CONDITION)")
    pcv1.add_argument("--context-hops", type=int, default=2,
                      help="BFS hops for neighborhood context (default 2)")
    pcv1.add_argument("--format", default="text", choices=["text", "json"])
    pcv1.add_argument("--report",  metavar="PATH", default=None,
                      help="write full JSON report to this file")
    pcv1.add_argument("--graphml", metavar="PATH", default=None,
                      help="write bipartite match GraphML to this file")
    pcv1.add_argument("--kg",      metavar="PATH", default=None,
                      help="write full model knowledge-graph GraphML to this file")
    pcv1.add_argument("--out-turtle", metavar="PATH", default=None,
                      help="write round-tripped model Turtle to this file")
    pcv1.add_argument("--ontology-graphml", metavar="PATH", default=None,
                      help="write ontology-diff GraphML to this file")
    pcv1.add_argument("--rvtm", metavar="PATH", default=None,
                      help="write the Requirements Verification & Traceability "
                           "Matrix (RVTM) to this CSV file")
    pcv1.add_argument("--rvtm-graphml", metavar="PATH", default=None,
                      help="write the RVTM traceability graph to this GraphML file")
    pcv1.add_argument("--template", default="IREB-Rupp")
    pcv1.add_argument("--backend",  default="rules", choices=["rules", "spacy", "bert"])
    pcv1.add_argument("--model",    default=None)
    pcv1.set_defaults(func=cmd_compare_v1)

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
