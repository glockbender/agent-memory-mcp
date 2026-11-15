#!/usr/bin/env python3
"""Audit frozen retrieval rows and evidence labels without an embedding model."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_source(root, relative):
    path = PurePosixPath(relative)
    require(path.parts and not path.is_absolute() and ".." not in path.parts,
            f"Invalid corpus path: {relative}")
    resolved = (root / relative).resolve(strict=True)
    require(resolved.is_relative_to(root), f"Source escapes root: {relative}")
    return resolved


def validate_rows(rows, questions, corpus, required_methods=None):
    methods = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["method"], row["id"], row["language"])
        require(key not in seen, f"Duplicate row: {key}")
        seen.add(key)
        require(row["id"] in questions and row["language"] in ("ru", "en"),
                f"Unknown question/language: {key}")
        question = questions[row["id"]]
        require(row["query"] == question[row["language"]], f"Changed query: {key}")
        require(row["primary_source"] == question["primary_source"], f"Changed label: {key}")
        hits = row["result"]["results"]
        require(len(hits) <= 5, f"Unexpected result budget: {key}")
        require(all(hit["path"] in corpus for hit in hits), f"Unknown result source: {key}")
        paths = [hit["path"] for hit in hits]
        rank = paths.index(question["primary_source"]) + 1 if question["primary_source"] in paths else None
        require(rank == row["primary_rank"], f"Incorrect stored rank: {key}")
        methods[row["method"]].append(row)
    expected = {(qid, lang) for qid in questions for lang in ("ru", "en")}
    for method, entries in methods.items():
        require({(r["id"], r["language"]) for r in entries} == expected,
                f"Incomplete method: {method}")
    if required_methods is not None:
        require(set(methods) == set(required_methods), "Incomplete or unexpected method set")
    return methods


def metrics(rows, accepted=None):
    ranks = []
    for row in rows:
        if accepted is None:
            rank = row["primary_rank"]
        else:
            rank = next((i + 1 for i, hit in enumerate(row["result"]["results"])
                         if hit["path"] in accepted[row["id"]]), None)
        ranks.append(rank)
    return {"queries": len(rows), "hit1": sum(r == 1 for r in ranks),
            "hit5": sum(r is not None for r in ranks),
            "mrr5": sum(1 / r for r in ranks if r) / len(rows)}


def replay_file_search(source_root, manifest, question, language):
    totals = defaultdict(lambda: [0, 0])
    paths = [str(source_root / item["path"]) for item in manifest["files"]]
    for term in question["rg_terms"][language]:
        run = subprocess.run(["rg", "--ignore-case", "--count", "--", term, *paths],
                             capture_output=True, text=True, timeout=30)
        require(run.returncode in (0, 1), f"rg failed: {run.stderr}")
        for line in run.stdout.splitlines():
            absolute, count = line.rsplit(":", 1)
            relative = str(Path(absolute).relative_to(source_root))
            totals[relative][0] += 1
            totals[relative][1] += int(count)
    ranked = sorted(totals, key=lambda p: (-totals[p][0], -totals[p][1], p))[:5]
    return [{"path": p, "terms_matched": totals[p][0], "lines_matched": totals[p][1]}
            for p in ranked]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    manifest = json.loads((args.dataset_dir / "corpus.json").read_text())
    original = json.loads((args.dataset_dir / "questions.json").read_text())["questions"]
    require(original, "Empty question set")
    require(len({q["id"] for q in original}) == len(original), "Duplicate question IDs")
    questions = {q["id"]: q for q in original}
    corpus = {item["path"] for item in manifest["files"]}
    require(len(corpus) == len(manifest["files"]), "Duplicate corpus paths")
    for item in manifest["files"]:
        require(sha256(checked_source(root, item["path"])) == item["sha256"],
                f"Source hash changed: {item['path']}")
    require(all(q["primary_source"] in corpus for q in original), "Missing primary source")
    rows = []
    for name in ("retrieval-results.jsonl", "semantic-results.jsonl"):
        rows.extend(json.loads(line) for line in (args.dataset_dir / name).read_text().splitlines())
    methods = validate_rows(rows, questions, corpus, {
        "agent-memory-keyword", "rg-terms", "semantic-hybrid-raw",
        "semantic-hybrid-prefixed", "vector-diagnostic-raw", "vector-diagnostic-prefixed"})
    published = json.loads((args.dataset_dir / "metrics.json").read_text())
    semantic_metrics = json.loads((args.dataset_dir / "semantic-metrics.json").read_text())
    require(sha256(args.dataset_dir / "questions.json") == published["frozen_questions_sha256"]
            == semantic_metrics["frozen_questions_sha256"], "Frozen question hash changed")
    published.update(semantic_metrics["results"])
    for method, entries in methods.items():
        for group in ("all", "ru", "en"):
            subset = entries if group == "all" else [r for r in entries if r["language"] == group]
            calculated = metrics(subset)
            saved = published[method][group]
            require(calculated["queries"] == saved["queries"], f"Published count differs: {method}/{group}")
            for metric in ("hit1", "hit5", "mrr5"):
                value = calculated[metric] / len(subset) if metric != "mrr5" else calculated[metric]
                require(abs(value - saved[metric]) < 1e-12, f"Published metric differs: {method}/{group}/{metric}")
    judgments = json.loads(args.judgments.read_text())
    needs = {entry["id"]: entry for entry in judgments["needs"]}
    require(set(needs) == set(questions) and len(needs) == len(judgments["needs"]),
            "Incomplete or duplicate audited needs")
    accepted = {}
    for qid, need in needs.items():
        accepted[qid] = set()
        for evidence in need["positive_sources"]:
            path = evidence["path"]
            require(path in corpus, f"Judgment source outside corpus: {path}")
            text = checked_source(root, path).read_text()
            require(evidence["quote"] and evidence["quote"] in text,
                    f"Missing source evidence: {qid}/{path}")
            accepted[qid].add(path)
    passage_counts = Counter()
    raw_lookup = {(r["id"], r["language"]): r for r in methods["semantic-hybrid-raw"]}
    seen_passages = set()
    for judgment in judgments["primary_passages"]:
        key = (judgment["id"], judgment["language"])
        require(key not in seen_passages, f"Duplicate passage judgment: {key}")
        seen_passages.add(key)
        row = raw_lookup[key]
        require(row["primary_rank"] is not None, f"No primary passage: {key}")
        hit = row["result"]["results"][row["primary_rank"] - 1]
        require(hit["id"] == judgment["chunk_id"], f"Changed passage ID: {key}")
        require(sha256_text(hit["evaluated_chunk"]) == judgment["chunk_sha256"],
                f"Changed passage text: {key}")
        require(judgment["grade"] in (0, 1, 2) and judgment["reason"],
                f"Invalid passage grade: {key}")
        passage_counts[str(judgment["grade"])] += 1
    require(seen_passages == {key for key, row in raw_lookup.items() if row["primary_rank"]},
            "Incomplete primary passage review")
    for row in methods["rg-terms"]:
        question = questions[row["id"]]
        require(row["result"]["terms"] == question["rg_terms"][row["language"]],
                "File-search terms changed")
        require(replay_file_search(root, manifest, question, row["language"])
                == row["result"]["results"], f"File-search replay differs: {row['id']}/{row['language']}")
    report = {"status": "passed", "corpus_files": len(corpus), "needs": len(questions),
              "frozen_questions_sha256": sha256(args.dataset_dir / "questions.json"),
              "judgments_sha256": sha256(args.judgments),
              "checks": ["source hashes", "source coverage", "query/label/rank integrity",
                         "complete paired rows", "positive-source evidence quotes",
                         "published metrics independently recomputed",
                         "all primary hit passages reviewed", "all file-search runs replayed"],
              "methods": {}, "primary_passage_grades": dict(passage_counts),
              "warning": "Audited positive sources are not exhaustive relevance labels; passage grades are agent judgments, not human gold labels."}
    for method, entries in methods.items():
        report["methods"][method] = {
            "original": metrics(entries), "audited_positive_sources": metrics(entries, accepted),
            "language": {lang: metrics([r for r in entries if r["language"] == lang], accepted)
                         for lang in ("ru", "en")}}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == "__main__":
    main()
