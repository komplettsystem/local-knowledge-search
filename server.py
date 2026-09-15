#!/usr/bin/env python3
"""Local hybrid (BM25 + vector) search MCP server. Pure in-process math,
no LLM calls, no network calls, no Docker. Rebuild the index with
build_index.py after the mirror changes; this process just serves queries
against whatever index is on disk at startup.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from fastembed import TextEmbedding
from mcp.server.fastmcp import FastMCP

from lib import load_json, tokenize, BM25, INDEX_DIR

RRF_K = 60
TOP_N_PER_METHOD = 30
N_RARE_TERMS = 3

# Verdict thresholds. STRONG requires either signal to be decisive on its
# own; WEAK requires both to be decisively low; anything else is MODERATE -
# the two cheap signals don't always resolve "topically adjacent but wrong"
# from "genuinely relevant with little shared vocabulary" (see README), so
# that ambiguous zone is reported as such rather than forced to a verdict
# the numbers don't support.
#
# Z_STRONG=3.0 was picked for margin against a *measured* confirmed miss
# (max z 2.61 across its full returned set, on a regression case - a
# query about an entity absent from the whole corpus that still
# lands on a topically-adjacent chunk), not fitted to split it from the
# confirmed hits (which measured max z >= 3.55). A miss mislabeled STRONG is
# the exact failure this tool exists to prevent; a hit mislabeled MODERATE
# costs little, since MODERATE still shows every number and the full chunk.
Z_STRONG = 3.0
Z_WEAK = 1.0
COVERAGE_STRONG = 0.6
COVERAGE_WEAK = 0.25

# A query term with zero corpus documents is close to proof the query names
# something the corpus never covers - it must not be outvoted by common
# terms that happen to match. See docs/relevance-signal-followup.md #1:
# measured absent_mass 0.577 on a confirmed miss whose coverage (42%) and z
# (2.4) otherwise landed in the MODERATE zone. A semantic-only paraphrase
# hit measured 0.462 - close enough to the 0.5 gate to be worth watching
# (see the follow-up doc's status section), but not yet crossing it.
ABSENT_MASS_WEAK = 0.5

mcp = FastMCP("local-knowledge-search")

_state: dict = {}


def _load_index():
    records = load_json("records.json")
    if not records:
        raise RuntimeError(
            "No index found. Run build_index.py first: "
            "~/.local-search/venv/bin/python3 ~/.local-search/build_index.py"
        )
    embeddings = np.load(INDEX_DIR / "embeddings.npy")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    bm25 = BM25([tokenize(r["text"]) for r in records])
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    _state.update(records=records, embeddings=embeddings, bm25=bm25, embedder=embedder)


def _verdict(z_best: float, coverage_best: float, absent_mass: float) -> str:
    if absent_mass >= ABSENT_MASS_WEAK:
        return "WEAK"
    if z_best >= Z_STRONG or coverage_best >= COVERAGE_STRONG:
        return "STRONG"
    if z_best < Z_WEAK and coverage_best < COVERAGE_WEAK:
        return "WEAK"
    return "MODERATE"


def _search(query: str, top_k: int) -> dict:
    records = _state["records"]
    bm25 = _state["bm25"]
    bm25_scores = bm25.scores_for_query(tokenize(query))
    bm25_ranked = sorted(range(len(records)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_N_PER_METHOD]

    query_vec = next(iter(_state["embedder"].embed([query])))
    query_vec = np.asarray(query_vec, dtype=np.float32)
    query_vec = query_vec / max(np.linalg.norm(query_vec), 1e-9)
    sims = _state["embeddings"] @ query_vec
    vec_ranked = sorted(range(len(records)), key=lambda i: sims[i], reverse=True)[:TOP_N_PER_METHOD]

    # Query-conditional background: how a random chunk in this corpus scores
    # against this query, so a hit's cosine can be read as "how far above
    # background" rather than as a meaningless absolute number.
    bg_mean = float(sims.mean())
    bg_std = float(sims.std())
    if bg_std < 1e-9:
        bg_std = 1e-9

    rrf_scores: dict[int, float] = {}
    for rank, idx in enumerate(bm25_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(vec_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

    best = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    # Distinct query terms weighted by corpus IDF. Coverage is reported as
    # the fraction of this IDF *mass* that literally appears in a chunk -
    # not a plain k/n count - so that matching one rare term (e.g. a proper
    # noun) counts for more than matching several common ones, and partial
    # overlap on mid-rarity terms isn't invisible just because it misses the
    # single rarest term. A handful of rarest terms are also picked out
    # purely for a human-readable display label (tie-broken on the term
    # itself so the label is reproducible run to run).
    distinct_terms = sorted(set(tokenize(query)))
    term_idf = {t: bm25.idf_for_term(t) for t in distinct_terms}
    total_idf_mass = sum(term_idf.values()) or 1e-9
    rare_terms = sorted(distinct_terms, key=lambda t: (-term_idf[t], t))[:N_RARE_TERMS]

    # Terms with zero corpus documents, and the share of the query's IDF
    # mass they hold - corpus-wide absence (not per-chunk absence), so this
    # is computed once here rather than per result below.
    absent_terms = [t for t in distinct_terms if t not in bm25.idf]
    absent_mass = sum(term_idf[t] for t in absent_terms) / total_idf_mass

    results = []
    for rank, (idx, score) in enumerate(best, start=1):
        r = records[idx]
        tf = bm25.term_freqs[idx]
        matched_terms = [t for t in distinct_terms if t in tf]
        coverage = sum(term_idf[t] for t in matched_terms) / total_idf_mass
        matched_rare = [t for t in rare_terms if t in tf]
        z = (float(sims[idx]) - bg_mean) / bg_std
        results.append({
            "path": r["path"],
            "chunk_index": r["chunk_index"],
            "content": r["text"],
            "score": round(score, 5),
            "rank": rank,
            "bm25": round(float(bm25_scores[idx]), 4),
            "cos": round(float(sims[idx]), 4),
            "z_raw": z,
            "z": round(z, 2),
            "coverage_raw": coverage,
            "coverage": round(coverage, 3),
            "rare_terms_matched": matched_rare,
            "rare_terms_total": rare_terms,
        })

    if results:
        # The verdict looks at the best signal anywhere in the returned set,
        # not rank 1's - RRF rank is a fusion of both methods and is not
        # guaranteed to put the highest-z or highest-coverage chunk first.
        # Tried restricting both to the top-ranked chunk instead (see
        # docs/relevance-signal-followup.md #2); reverted after measuring it
        # against 25 real queries - it demonstrably drops real hits from
        # STRONG to MODERATE when fusion buries the actually-relevant chunk
        # below rank 1 (a query about disputing a parking ticket ranked the
        # matching procedure doc 6th, at z=3.13, behind an unrelated
        # doc at rank 1, z=2.45), with no observed case of the theorized
        # opposite failure (a spurious non-top-ranked z manufacturing a
        # false STRONG).
        z_best = max(r["z_raw"] for r in results)
        coverage_best = max(r["coverage_raw"] for r in results)
        verdict = _verdict(z_best, coverage_best, absent_mass)
    else:
        z_best = 0.0
        coverage_best = 0.0
        verdict = "WEAK"

    return {
        "results": results,
        "verdict": verdict,
        "z_best": round(z_best, 2),
        "coverage_best": round(coverage_best, 3),
        "rare_terms": rare_terms,
        "absent_terms": absent_terms,
        "absent_mass": round(absent_mass, 3),
        "bm25_spread": round(float(np.std([r["bm25"] for r in results])), 4) if results else 0.0,
        "cos_spread": round(float(np.std([r["cos"] for r in results])), 4) if results else 0.0,
    }


@mcp.tool()
def search_local_docs(query: str, top_k: int = 8) -> str:
    """Search the user's own personal notes, project docs, and work history
    (markdown files: strategy docs, session handovers, technical research,
    decisions made, past incidents, etc.).

    Call this FIRST - before web search, before any other knowledge-base or
    document-search tool - for any question that touches the user's own
    projects, prior decisions, past work, or anything they or their team
    wrote down. Only fall back to web search if this returns nothing
    relevant, or the question is clearly about public/external information
    (news, public documentation, general facts) that could not possibly be
    in the user's own notes.

    Hybrid BM25 + vector similarity search, pure local computation - no LLM
    calls, no network calls, single-digit-ms per query. Returns raw ranked
    chunks annotated with relevance signals (IDF-weighted term coverage,
    background-relative z-score) for you to read and judge yourself -
    nothing is filtered out, including weak matches, so a "WEAK" or
    "MODERATE" verdict means judge the chunks skeptically, not that
    anything was hidden."""
    out = _search(query, top_k)
    results = out["results"]
    if not results:
        return "No results."

    rare_terms = out["rare_terms"]
    rare_label = ", ".join(rare_terms) if rare_terms else "(no distinctive terms in query)"
    header = (
        f'query: "{query}"\n'
        f'relevance: {out["verdict"]} — best-in-set z {out["z_best"]:+.1f}σ above background, '
        f'best-in-set IDF-weighted coverage {out["coverage_best"]*100:.0f}% '
        f'(rarest query terms: [{rare_label}])\n'
        f'spread across returned set: bm25 σ={out["bm25_spread"]}, cos σ={out["cos_spread"]}'
    )
    if out["absent_terms"]:
        header += (
            f'\n{len(out["absent_terms"])} query term(s) absent from the entire corpus: '
            f'{out["absent_terms"]} ({out["absent_mass"]*100:.0f}% of query IDF mass)'
        )
    if out["verdict"] == "WEAK":
        header += "\n(corpus likely contains little or nothing on this topic - all results still shown below)"
    elif out["verdict"] == "MODERATE":
        header += "\n(signals are ambiguous - could be a real but loosely-worded match, or a topically adjacent false positive; read the chunks below before trusting them)"

    lines = [header]
    for r in results:
        rare_str = f'{len(r["rare_terms_matched"])}/{len(rare_terms)} {rare_terms}' if rare_terms else "n/a"
        lines.append(
            f"### {r['path']} (chunk {r['chunk_index']})\n"
            f'    rank {r["rank"]} | z {r["z"]:+.1f}σ | coverage {r["coverage"]*100:.0f}% | '
            f'rare_terms {rare_str} | bm25 {r["bm25"]} | cos {r["cos"]} | fused {r["score"]}\n'
            f"{r['content']}"
        )
    return "\n\n---\n\n".join(lines)


if __name__ == "__main__":
    _load_index()
    mcp.run(transport="stdio")
