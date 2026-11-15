# Live document search regression test

This opt-in macOS test exercises a compiled binary against a real local Ollama
embedding model and externally supplied documents. It is kept on the fork's
`codex/live-search-regression` branch, which combines the discovery-safety,
Ollama-model-selection and directory-symlink changes. It is separate from the
focused upstream directory-symlink PR and does not run in ordinary CI.

The script uses the Python standard library and an existing Ollama installation.
It starts its own loopback server with a temporary home, model store and database.
Session tracking, memory, background capture and hosted embedding providers are
disabled. Existing Ollama instances, model stores and agent settings are untouched.

## Inputs and execution

Build the binary from this branch, then run:

```sh
go build -o /tmp/agent-memory-live-test ./cmd/agent-memory-mcp
python3 scripts/test-live-document-search.py \
  --binary /tmp/agent-memory-live-test \
  --source-root /path/to/source-repository \
  --corpus-manifest /path/to/corpus.json \
  --questions /path/to/questions.json \
  --baseline /path/to/semantic-results.jsonl \
  --output-dir /path/to/new-results-directory
```

The manifest has a `files` array with relative `path` and `sha256` fields. Question
input has a `questions` array with `id`, `ru`, `en` and `primary_source` fields.
Baseline JSONL rows have `method: "semantic-hybrid-raw"`, `id`, `language` and
`primary_rank` (1–5 or null). Each question needs one baseline row per language.
Supply trusted, reviewed material; documents and results are never uploaded.

By default the test downloads `embeddinggemma:300m`, about 622 MB, pinned to its
recorded model digest and 768 dimensions. Changing the model requires matching
`--model`, `--model-digest`, `--dimension` and a suitable baseline. Source hashes
are checked before downloading. The output directory must be new.

## Assertions and retained evidence

The test fails if a previously successful source hit is lost, if retrieval falls
back from the selected Ollama model, or if any expected file/vector is missing.
It replaces the copied source directories with links and requires identical file
and chunk records plus identical ranked paths for every question. A broken
configured link must refuse indexing without changing files, chunks or metadata.
It also checks that no memory database was created.

The output contains ordinary/link result rows and `summary.json` with input and
binary hashes, model identity, counts, assertion results and cleanup details.
These outputs may identify private sources; keep them in the source repository's
research area, outside this public fork.

The owned Ollama process is stopped even on test failure. Downloaded model files,
database files and corpus copies are removed, then the empty temporary directory
tree is moved to macOS Trash. If owned files reach 1 GB, removal stops for manual
confirmation and the summary reports their location and size. The input binary
and retained output files are not removed by the test.

## Audit the baseline before changing ranking

Run the model-free integrity and evidence audit separately:

```sh
python3 -B scripts/test_audit_search_baseline.py
python3 scripts/audit-search-baseline.py \
  --dataset-dir /path/to/frozen-trial \
  --source-root /path/to/source-repository \
  --judgments /path/to/judgments.json \
  --output /path/to/audit.json
```

The audit checks all six historical methods, paired questions, source hashes,
actual ranks and published metrics. It replays every saved file-search query.
Judgments list each need's evidence-backed `positive_sources` and separately
grade its returned primary-source chunk. They are sparse positive labels and
agent assessments, not exhaustive human ground truth. Keep historical
single-source metrics alongside the audited metrics; a different useful source
is not automatically a failure. Full-chunk judgments do not establish that the
shorter delivered snippet contains the answer.

## Trace and compare ranking on the live index

Build the opt-in diagnostic test binary from this branch:

```sh
go test -tags=corpus -c -o /tmp/agent-memory-ranking-probe ./internal/rag
```

Add these flags to the live test command:

```sh
--compare-semantic \
--ranking-probe /tmp/agent-memory-ranking-probe \
--ranking-input /path/to/diagnostic-input.json
```

The input is an array of `id`, `language`, `query`, `accepted_sources` and `kind`
objects. Empty accepted-source lists mark no-answer controls, which are retained
but excluded from positive-source hit metrics. The probe holds native candidate
budgets fixed while comparing ranking, then separately tests a larger semantic
pool and removal of recency boosts. It records candidate ranks, scores and
actual native search responses. The selected semantic mode must find an
accepted source within five for each positive input. The probe skips when its
explicit isolated configuration is absent.

`--compare-semantic` also checks selection through the actual CLI configuration
on all original questions. It writes `semantic.jsonl` and summary counts, then
restores the default profile before the symlink checks. The diagnostic output
includes private query vectors and source chunks; keep it with local research
results. Models and temporary databases are removed as in the ordinary run.
