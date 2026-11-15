#!/usr/bin/env python3
"""Opt-in live-model regression test; all supplied corpus data stays local."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import signal
import socket
import sqlite3
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ollama", default="ollama")
    parser.add_argument("--model", default="embeddinggemma:300m")
    parser.add_argument("--model-digest", default="85462619ee721b466c5927d109d4cb765861907d5417b9109caebc4e614679f1")
    parser.add_argument("--dimension", type=int, default=768)
    args = parser.parse_args()
    args.binary = args.binary.resolve(strict=True)
    args.source_root = args.source_root.resolve(strict=True)
    manifest = json.loads(args.corpus_manifest.read_text())
    questions = json.loads(args.questions.read_text())["questions"]
    baseline = {}
    for line in args.baseline.read_text().splitlines():
        row = json.loads(line)
        if row["method"] == "semantic-hybrid-raw":
            baseline[(row["id"], row["language"])] = row["primary_rank"]
    require(len(baseline) == len(questions) * 2, "Incomplete baseline")
    for item in manifest["files"]:
        relative = PurePosixPath(item["path"])
        require(not relative.is_absolute() and relative.parts
                and ".." not in relative.parts and "\\" not in item["path"],
                f"Invalid relative corpus path: {item['path']}")
        source = (args.source_root / item["path"]).resolve(strict=True)
        require(source.is_relative_to(args.source_root), "Source escapes source root")
        require(digest(source) == item["sha256"], f"Source changed: {item['path']}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    temporary = Path(tempfile.mkdtemp(prefix="agent-memory-live-"))
    home, corpus, models = (temporary / name for name in ("home", "corpus", "models"))
    for directory in (home, corpus, models):
        directory.mkdir()
    env = {"HOME": str(home), "PATH": "/opt/homebrew/bin:/usr/bin:/bin"}
    result = {"status": "failed", "model": args.model,
              "questions_sha256": digest(args.questions),
              "corpus_manifest_sha256": digest(args.corpus_manifest),
              "baseline_sha256": digest(args.baseline),
              "binary_sha256": digest(args.binary), "checks": []}
    server = None
    log = None
    url = None

    def api(path, payload=None, timeout=60):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(url + path, data=data,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    try:
        for item in manifest["files"]:
            target = corpus / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.source_root / item["path"], target)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        log = (temporary / "ollama.log").open("w")
        server = subprocess.Popen([args.ollama, "serve"], cwd=temporary,
                                  env=dict(env, OLLAMA_HOST=url, OLLAMA_MODELS=str(models)),
                                  stdout=log, stderr=log, start_new_session=True)
        deadline = time.monotonic() + 30
        while True:
            require(server.poll() is None, "Isolated Ollama exited")
            try:
                api("/api/tags", timeout=1)
                break
            except (OSError, urllib.error.URLError):
                require(time.monotonic() < deadline, "Ollama startup timed out")
                time.sleep(0.2)
        print("Downloading the isolated model", flush=True)
        api("/api/pull", {"model": args.model, "stream": False}, timeout=600)
        result["model_metadata"] = api("/api/tags")
        require(any(model["name"] == args.model and model["digest"] == args.model_digest
                    for model in result["model_metadata"]["models"]),
                "Downloaded model does not match the pinned digest")
        result["ollama_version"] = api("/api/version")

        settings = {
            "MCP_ROOT": str(corpus), "MCP_DATA_PATH": str(temporary / "data"),
            "MCP_INDEX_DIRS": ",".join(sorted({Path(f["path"]).parts[0]
                                             for f in manifest["files"]})),
            "MCP_EMBEDDING_MODE": "local-only", "OLLAMA_BASE_URL": url,
            "OLLAMA_EMBEDDING_MODEL": args.model,
            "MCP_EMBEDDING_DIMENSION": str(args.dimension),
            "MCP_RETRIEVAL_STRICT": "true", "MCP_HTTP_MODE": "stdio",
            "MCP_RAG_ENABLED": "true", "MCP_RAG_MAX_CHUNKS_PER_DOC": "1",
            "MCP_CHUNK_SIZE": "2000", "MCP_CHUNK_OVERLAP": "200",
        }
        for key in ("MEMORY_ENABLED", "SESSION_TRACKING_ENABLED", "COMPILE_BACKGROUND",
                    "STEWARD_ENABLED", "ARCHIVE_SWEEP_ENABLED", "TRIPLE_EXTRACTOR_ENABLED",
                    "SEDIMENT_ENABLED", "RAG_AUTO_INDEX", "RAG_FILE_WATCHER",
                    "RAG_KEYWORD_FALLBACK", "RERANK_ENABLED"):
            settings["MCP_" + key] = "false"
        config = temporary / "config.env"
        config.write_text("".join(f"{key}={value}\n" for key, value in settings.items()))

        def command(action, *extra, success=True):
            completed = subprocess.run([str(args.binary), action, "--config", str(config),
                                        *extra], cwd=temporary, env=env, text=True,
                                       capture_output=True, timeout=300)
            require(not success or completed.returncode == 0,
                    f"{action} failed: {completed.stderr[-4000:]}")
            return completed

        database = temporary / "data" / "rag-index" / "vectors.db"

        def state():
            with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
                files = connection.execute("SELECT * FROM indexed_files ORDER BY file_path").fetchall()
                chunks = connection.execute("SELECT * FROM chunks ORDER BY id").fetchall()
                metadata = connection.execute("SELECT * FROM index_metadata ORDER BY key").fetchall()
            require({row[0] for row in files} == {f["path"] for f in manifest["files"]},
                    "Indexed file coverage changed")
            require(chunks, "No indexed chunks")
            require(all(len(row[-1]) == args.dimension * 4 for row in chunks),
                    "Missing or wrong-size vectors")
            return {"files": len(files), "chunks": len(chunks),
                    "content_hash": hashlib.sha256(repr((files, chunks)).encode()).hexdigest(),
                    "metadata": dict(metadata)}

        order = [(q, lang) for q in questions for lang in ("ru", "en")]
        random.Random(20261115).shuffle(order)

        def evaluate(profile):
            rows = []
            with (args.output_dir / f"{profile}.jsonl").open("w") as output:
                for question, language in order:
                    response = json.loads(command("search", "-json", "-limit", "5",
                                                  question[language]).stdout)
                    retrieval = response.get("retrieval", {})
                    require(retrieval.get("embedding_provider") == "ollama/" + args.model,
                            f"Unexpected provider: {retrieval}")
                    require(not retrieval.get("degraded") and not retrieval.get("keyword_only"),
                            f"Degraded search: {retrieval}")
                    paths = [r["path"] for r in response["results"]]
                    rank = paths.index(question["primary_source"]) + 1 if question["primary_source"] in paths else None
                    row = {"id": question["id"], "language": language,
                           "primary_rank": rank, "paths": paths, "retrieval": retrieval}
                    rows.append(row)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
            return rows

        print("Indexing and evaluating ordinary directories", flush=True)
        command("index")
        result["initial_state"] = state()
        ordinary = evaluate("ordinary")
        targets = temporary / "targets"
        targets.mkdir()
        for name in settings["MCP_INDEX_DIRS"].split(","):
            target = targets / name
            (corpus / name).rename(target)
            (corpus / name).symlink_to(target, target_is_directory=True)
        print("Reindexing and evaluating directory symlinks", flush=True)
        command("index")
        result["linked_state"] = state()
        linked = evaluate("linked")
        require(result["initial_state"]["content_hash"] == result["linked_state"]["content_hash"],
                "Directory replacement changed indexed records")
        require(ordinary == linked, "Directory replacement changed ranked results")
        lost = [r for r in ordinary if baseline[(r["id"], r["language"])] and not r["primary_rank"]]
        result["baseline_lost_hits"] = lost
        for language in ("ru", "en"):
            subset = [r for r in ordinary if r["language"] == language]
            result[language] = {"hit5": sum(r["primary_rank"] is not None for r in subset),
                                "hit1": sum(r["primary_rank"] == 1 for r in subset),
                                "total": len(subset)}
        require(not lost, "Lost source hits compared with frozen baseline")
        result["checks"].extend(["same corpus hashes", "same frozen questions",
                                  "native selected Ollama model; no fallback",
                                  "stable records and ranked results through directory symlinks"])

        first = settings["MCP_INDEX_DIRS"].split(",")[0]
        (targets / first).rename(targets / (first + "-unavailable"))
        before = state()
        failure = command("index", success=False)
        require(failure.returncode != 0, "Broken configured symlink was accepted")
        require(before == state(), "Failed discovery changed index state")
        result["checks"].append("broken configured symlink refuses indexing without state changes")
        require(not list(temporary.rglob("memories.db")), "Unexpected memory database")
        result["checks"].append("no memory database")
        result["status"] = "passed"
        print("All live regression checks passed", flush=True)
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        result["cleanup"] = {"temporary_root": str(temporary)}
        try:
            if server is not None:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait(timeout=5)
                result["server_exit_code"] = server.returncode
            if log is not None:
                log.close()
            owned_files = [p for p in temporary.rglob("*") if p.is_file() and not p.is_symlink()]
            allocated = sum(p.stat().st_blocks * 512 for p in owned_files)
            logical = sum(p.stat().st_size for p in owned_files)
            result["cleanup"].update(files=len(owned_files), allocated_bytes=allocated,
                                     logical_bytes=logical)
            require(logical < 1_000_000_000, "Cleanup exceeds 1 GB; manual confirmation required")
            for path in temporary.rglob("*"):
                if path.is_symlink() or path.is_file():
                    path.unlink()
            subprocess.run(["/usr/bin/trash", str(temporary)], check=True)
            result["cleanup"]["temporary_root_absent"] = not temporary.exists()
        except Exception as error:
            result["status"] = "failed"
            result["cleanup"]["error"] = str(error)
            raise
        finally:
            (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
