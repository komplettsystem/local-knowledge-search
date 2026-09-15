# local-knowledge-search

> **Status:** published for reference. Issues and pull requests are closed; this is a personal tool shared as-is, not a maintained project.

A small, self-contained hybrid (BM25 + vector) search engine over a folder of
markdown files, exposed as an MCP server. No Docker, no database, no LLM
calls and no network calls at query time — everything runs in one local
Python process, in single-digit milliseconds per query.

## Why this exists

It's a drop-in alternative to running a full retrieval stack (e.g.
[Onyx](https://onyx.app)) just to let Claude Code / Claude Desktop / Cowork
search a personal knowledge base. Onyx's own retrieval pipeline calls an LLM
multiple times per search (query expansion, then per-chunk relevance
classification) to rank results — measured at 10–60+ seconds per query
depending on provider. For a modest, mostly-static folder of notes, that
LLM-based reranking is unnecessary: classic BM25 + embedding similarity,
combined with reciprocal rank fusion, gets comparably relevant results
without any of it. This tool returns raw ranked chunks; reading them and
writing the actual answer is left entirely to whichever Claude session calls
it — the same way it would use any other search tool.

This project has **no dependency on Onyx, Docker, or any particular sync
mechanism**. It indexes whatever folder of markdown files you point it at.

## How it works

- **Chunking** (`lib.py`): splits each file on `#`/`##` headings, then
  further splits any section longer than ~1200 characters into overlapping
  ~800-character windows.
- **Keyword search**: a from-scratch BM25Okapi implementation (`lib.py`,
  `BM25` class) — no external dependency for this part.
- **Vector search**: each chunk is embedded with `BAAI/bge-small-en-v1.5` via
  [`fastembed`](https://github.com/qdrant/fastembed) (ONNX runtime, CPU-only,
  no GPU or external model server required). Query-time similarity is plain
  cosine similarity against the cached embedding matrix.
- **Combining the two**: [reciprocal rank fusion](https://en.wikipedia.org/wiki/Learning_to_rank#Reciprocal_rank_fusion)
  (`k=60`) over the top 30 hits from each method — no score normalization
  needed, no tuning knobs.
- **Caching**: embeddings are cached on disk keyed by content hash
  (`index/embed_cache.npz`), so re-running the indexer after small edits only
  re-embeds what actually changed.

### Relevance signal

RRF's fused score is rank-only: rank 1 of a great match and rank 1 of the
least-bad chunk in a corpus that has nothing on the topic score identically
(`1/61 + 1/61 = 0.0328` either way). So the fused `score` field is kept for
backward compatibility but carries no relevance information on its own —
`search_local_docs` attaches three signals per result instead, computed from
data already produced during the search (no extra index artifacts, no
calibration constants, negligible added latency):

- **IDF-weighted term coverage** — what fraction of the query's IDF *mass*
  (not just a plain word count) literally appears in the chunk, using the
  same tokenization as the BM25 index. A proper noun or rare term absent
  from the entire corpus (not just this chunk) gets the maximum possible
  IDF, so a query about an entity the corpus has never seen shows near-zero
  coverage rather than falling back to a default that reads as "common."
- **Background-relative z-score** — cosine similarity is meaningless in
  absolute terms (unrelated chunk pairs in a topical corpus often sit at
  0.5–0.7), so each hit is reported as how many standard deviations above
  this *query's* similarity distribution over the whole corpus it lands.
  Self-calibrating: no stored threshold, adapts per query and per corpus.
- **Score spread across the returned set** — stdev of the raw BM25 and
  cosine scores (never the fused score, whose spread is near-constant by
  RRF's construction and carries no information) across the chunks actually
  returned, as a rough "is this ranking flat or does it have a real top hit"
  indicator.
- **Corpus-absent query terms** — the share of the query's IDF mass held by
  terms with zero matching documents anywhere in the corpus (not just the
  returned chunks). A term absent from the *whole corpus* is close to proof
  the query names something the corpus never covers, so it overrides `z`
  and coverage rather than being averaged against them; a term merely
  absent from one chunk carries no such weight and isn't part of this
  signal at all (see `docs/relevance-signal-followup.md`).

These are combined into an advisory `STRONG` / `MODERATE` / `WEAK` header,
but the raw numbers are always included and **nothing is ever filtered or
hidden** based on them — a silently dropped result is worse than a weak one
the caller can see is weak. `z_best`/`coverage_best` are each the max over
the whole returned set, not necessarily rank 1's — RRF's fused rank doesn't
guarantee the highest-z or highest-coverage chunk sorts first, and a real
hit can rank below an unrelated chunk if its literal vocabulary overlap is
weak (see `docs/relevance-signal-followup.md` for a measured case).

**Measured limitation, partially closed:** in a heterogeneous personal-notes
corpus, a query about a specific absent entity (e.g. a place name with no
matching content) can land a topically-adjacent chunk at a moderate z-score
and low-but-nonzero coverage — numerically close to a genuine semantic match
that shares little literal vocabulary with its best chunk. Measured on this
corpus: a query about an entity absent from the entire corpus (place name
with zero real content) reached best-in-set z = 2.4σ / coverage 42%, while a
genuine hit reached z = 3.9σ / coverage 81% — a real but not huge gap (about
1.5σ) for `z`/coverage alone to resolve. That specific case is now caught
earlier: the miss's identifying terms (the place name and a proper noun) are
absent from the whole corpus and hold 58% of the query's IDF mass, which
forces `WEAK` regardless of `z`/coverage (see the corpus-absent query terms
signal above). The residual gap remains for misses
that land on a topically-adjacent chunk *without* any query term being
wholly absent from the corpus — `z`/coverage are still the only signals for
that case, which is why the verdict keeps three tiers instead of two:
`STRONG` requires clearing the measured hit/miss gap with margin (`z >= 3.0`
or `coverage >= 60%`), `WEAK` requires both signals to be decisively low (or
the corpus-absent-terms gate above), and everything in between is reported
as `MODERATE` ("ambiguous — read the chunk yourself") rather than forced
toward a confident-sounding verdict the numbers don't actually support.

## Files

| File | Role |
|---|---|
| `lib.py` | Chunking, tokenizing, the BM25 implementation, corpus loading, index I/O helpers |
| `build_index.py` | Walks the content folder, chunks + embeds everything (incrementally), writes `index/records.json` and `index/embeddings.npy` |
| `server.py` | MCP stdio server exposing one tool, `search_local_docs(query, top_k=8)` |
| `test_relevance.py` | Plain-assertion tests: verdict logic (runs anywhere) plus two live-index tests that read their queries from environment variables (see the file header) and skip without them |
| `docs/relevance-signal-followup.md` | Design log for the relevance verdict: a measured miss, the diagnosis, the fix, and one change reverted on evidence |
| `requirements.txt` | `fastembed`, `mcp<2` (the classic `FastMCP` API — v2 renamed it to `MCPServer`), `numpy` |
| `index/` | Generated at build time — not committed (see `.gitignore`) |
| `venv/` | Generated at install time — not committed |

## Installation

```bash
git clone <this-repo> local-knowledge-search
cd local-knowledge-search
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Point it at your content (optional — defaults to `~/.knowledge-base-sync/mirror`, this
author's own setup):

```bash
export LOCAL_SEARCH_CONTENT_ROOT="/path/to/your/markdown/folder"
```

Build the index (first run downloads the ~130MB embedding model; re-runs are
fast — only changed chunks get re-embedded):

```bash
./venv/bin/python3 build_index.py
```

Run the tests:

```bash
./venv/bin/python3 test_relevance.py
```

### Wire it into Claude Code

Add to `~/.claude.json` (top-level `mcpServers`, or a project's local one):

```json
"local-knowledge-search": {
  "type": "stdio",
  "command": "/absolute/path/to/local-knowledge-search/venv/bin/python3",
  "args": ["/absolute/path/to/local-knowledge-search/server.py"]
}
```

### Wire it into Claude Desktop / Cowork

Add the same shape to `~/Library/Application Support/Claude/claude_desktop_config.json`'s
`mcpServers` (the `type` key isn't needed there):

```json
"local-knowledge-search": {
  "command": "/absolute/path/to/local-knowledge-search/venv/bin/python3",
  "args": ["/absolute/path/to/local-knowledge-search/server.py"]
}
```

Use absolute paths for both `command` and `args` — always use `command`/`args`
(stdio), never a `url` field, in `claude_desktop_config.json`: Claude Desktop
has been observed to silently wipe its entire `mcpServers` section when an
entry uses `url` instead of `command` (anthropics/claude-code#37286).

**Known quirk:** the Desktop app can rewrite `claude_desktop_config.json` from
its own in-memory state (e.g. on quit), which will silently drop an edit made
to the file while the app was running and never reloaded. Fully quit and
relaunch Claude Desktop shortly after editing this file, rather than leaving
it running for a long stretch first.

### Keeping the index fresh

Re-run `build_index.py` whenever your content changes. If it's already on a
schedule (cron, launchd, a CI job), just add this as one more step — it's
cheap (well under a second) when nothing changed. Example launchd-driven
shell step:

```bash
cd /absolute/path/to/local-knowledge-search && ./venv/bin/python3 build_index.py
```

## Performance (measured against a 357-file / 3,700-chunk personal notes
corpus)

| | |
|---|---|
| Cold index build (with model download) | ~220s (one-time) |
| Incremental re-index, no changes | ~0.2s |
| Incremental re-index, a few changed files | <1s |
| Query (index already loaded) | ~5.5ms warm median, both before and after adding relevance signals |

For comparison, the same corpus through Onyx's LLM-based reranking pipeline
measured 10–60+ seconds per query, depending on which LLM provider backed
the reranking step.

## Limitations

- No access control / multi-user support — this is a single-user, local tool.
- No connectors (Slack, Confluence, GitHub, etc.) — markdown files on disk
  only. If you need those, use a real retrieval platform (Onyx, Glean, etc.)
  instead; this tool intentionally does not try to replace that.
- Relevance quality depends entirely on BM25 + a small embedding model — no
  LLM-based reranking. For a large or heterogeneous corpus this may rank
  worse than an LLM-reranked pipeline; it was built for a personal-notes-scale
  corpus where that trade-off is worth the 1000x+ latency improvement.
