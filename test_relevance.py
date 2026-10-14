#!/usr/bin/env python3
"""Plain-assertion tests for the relevance verdict logic (no test framework
dependency, matching this project's minimal-deps philosophy).

Run: ./venv/bin/python3 test_relevance.py

Covers docs/relevance-signal-followup.md item #1 (implemented):
- absent-term IDF-mass gate forces WEAK regardless of z/coverage

Item #2 (restricting z_best/coverage_best to the top-ranked chunk) was
tried and reverted - see the status section in
docs/relevance-signal-followup.md and the comment in server.py's _search().
z_best/coverage_best remain independent maxima across the whole returned
set; test_z_and_coverage_best_are_independent_maxima below pins that down
so a future change doesn't silently reintroduce the top-ranked restriction.

The verdict tests are pure unit tests and run anywhere. The two live-index
tests depend on what is in *your* corpus, so their queries come from
environment variables and they are skipped when those are unset:

  LKS_MISS_QUERY         a query naming something your corpus never covers
  LKS_MISS_ABSENT_TERMS  comma-separated terms from it expected to be corpus-absent
  LKS_HIT_QUERY          a query your corpus clearly answers
  LKS_FUSION_QUERY       a query whose best-z chunk is not ranked first by RRF
"""
import os
import tempfile
from pathlib import Path

import lib
import server


class Skip(Exception):
    pass


def _env(*names):
    values = [os.environ.get(n, "").strip() for n in names]
    if not all(values):
        raise Skip(f"set {', '.join(names)} to run against your own index")
    return values


def test_verdict_absent_mass_gate_forces_weak():
    # High z and high coverage would normally be STRONG, but heavy absent
    # query-term mass must override that.
    assert server._verdict(z_best=5.0, coverage_best=0.9, absent_mass=0.7) == "WEAK"


def test_verdict_no_absent_terms_unaffected():
    # No corpus-absent terms: z and coverage decide, as before.
    assert server._verdict(z_best=5.0, coverage_best=0.9, absent_mass=0.0) == "STRONG"
    assert server._verdict(z_best=0.2, coverage_best=0.1, absent_mass=0.0) == "WEAK"
    assert server._verdict(z_best=2.0, coverage_best=0.4, absent_mass=0.0) == "MODERATE"


def test_verdict_any_absent_term_caps_at_moderate():
    # A query word the corpus never contains means the nearest chunk may match
    # only the query's ordinary words (e.g. "sourdough starter feeding"), so a
    # high z or coverage can't be read as STRONG. Label only: results unchanged.
    assert server._verdict(z_best=5.0, coverage_best=0.9, absent_mass=0.1) == "MODERATE"
    assert server._verdict(z_best=0.2, coverage_best=0.1, absent_mass=0.1) == "WEAK"


def test_verdict_z_alone_needs_some_coverage_for_strong():
    # Rare words from unrelated notes, every one present somewhere in the corpus:
    # the nearest chunk sits well above background but shares about a third of
    # the query's weight. See docs/relevance-signal-followup.md, "Coverage floor".
    assert server._verdict(z_best=4.4, coverage_best=0.39, absent_mass=0.0) == "MODERATE"
    assert server._verdict(z_best=4.4, coverage_best=0.45, absent_mass=0.0) == "STRONG"
    # The coverage path to STRONG is unchanged.
    assert server._verdict(z_best=0.5, coverage_best=0.6, absent_mass=0.0) == "STRONG"


def test_verdict_gate_boundary():
    assert server._verdict(z_best=5.0, coverage_best=0.9, absent_mass=0.5) == "WEAK"
    assert server._verdict(z_best=5.0, coverage_best=0.9, absent_mass=0.49) == "MODERATE"


def test_verdict_rerank_gate_caps_strong():
    # STRONG also needs the reranker's score for the top chunk to clear
    # RERANK_STRONG_MIN: a chunk on the right topic that doesn't hold the asked
    # fact scores well on z and coverage but low on the reranker.
    low, high = server.RERANK_STRONG_MIN - 1, server.RERANK_STRONG_MIN + 1
    assert server._verdict(z_best=5.0, coverage_best=0.9, absent_mass=0.0, rerank_top=low) == "MODERATE"
    assert server._verdict(z_best=5.0, coverage_best=0.9, absent_mass=0.0, rerank_top=high) == "STRONG"
    # Reranker off: unchanged.
    assert server._verdict(z_best=5.0, coverage_best=0.9, absent_mass=0.0, rerank_top=None) == "STRONG"
    # The gate only demotes STRONG; it never lifts or lowers the other tiers.
    assert server._verdict(z_best=0.2, coverage_best=0.1, absent_mass=0.0, rerank_top=high) == "WEAK"
    assert server._verdict(z_best=2.0, coverage_best=0.4, absent_mass=0.0, rerank_top=low) == "MODERATE"


def test_apply_rerank_reorders_and_cuts():
    results = [{"path": p} for p in "abcd"]
    out = server._apply_rerank(results, [0.1, 2.0, -1.0, 2.0], top_k=3)
    # Highest score first; ties keep the fused order.
    assert [r["path"] for r in out] == ["b", "d", "a"]
    assert [r["rank"] for r in out] == [1, 2, 3]
    assert [r["rerank"] for r in out] == [2.0, 2.0, 0.1]


def test_exclude_conf_skips_listed_folders():
    with tempfile.TemporaryDirectory() as d:
        conf = Path(d) / "exclude.conf"
        conf.write_text("# old notes\nnotes/retired/\n\n")
        prefixes = lib.load_excludes(conf)
        assert prefixes == ["notes/retired/"]
        assert lib.is_excluded("notes/retired/a.md", prefixes)
        assert not lib.is_excluded("notes/retired-plan.md", prefixes)
        assert not lib.is_excluded("notes/a.md", prefixes)
        assert lib.load_excludes(Path(d) / "missing.conf") == []


def test_live_index_confirmed_miss_is_weak():
    miss_query, absent, hit_query = _env("LKS_MISS_QUERY", "LKS_MISS_ABSENT_TERMS", "LKS_HIT_QUERY")
    server._load_index()

    out = server._search(miss_query, top_k=3)
    assert out["verdict"] == "WEAK", (
        f'expected WEAK for the confirmed-miss query, got {out["verdict"]} '
        f'(z_best={out["z_best"]}, coverage_best={out["coverage_best"]}, '
        f'absent_terms={out["absent_terms"]})'
    )
    for term in absent.lower().split(","):
        assert term.strip() in out["absent_terms"]
    assert len(out["results"]) == 3, "results must still be returned in full on a WEAK verdict"

    out_good = server._search(hit_query, top_k=3)
    assert out_good["verdict"] == "STRONG", (
        f'expected STRONG for the known-good query, got {out_good["verdict"]}'
    )


def test_z_and_coverage_best_are_independent_maxima():
    # Regression case for the reverted top-ranked-only approach: the
    # matching doc for this query ranks below 1st by RRF fusion, with the
    # highest z in the returned set. z_best must still pick it up.
    (fusion_query,) = _env("LKS_FUSION_QUERY")
    server._load_index()
    out = server._search(fusion_query, top_k=8)
    results = out["results"]
    assert out["z_best"] == round(max(r["z_raw"] for r in results), 2)
    assert out["coverage_best"] == round(max(r["coverage_raw"] for r in results), 3)
    assert results[0]["z_raw"] != max(r["z_raw"] for r in results), (
        "test fixture assumption broke: rank 1 is now also the max-z chunk, "
        "so this no longer exercises the independent-maxima behavior"
    )


if __name__ == "__main__":
    tests = [
        test_verdict_absent_mass_gate_forces_weak,
        test_verdict_no_absent_terms_unaffected,
        test_verdict_any_absent_term_caps_at_moderate,
        test_verdict_z_alone_needs_some_coverage_for_strong,
        test_verdict_gate_boundary,
        test_verdict_rerank_gate_caps_strong,
        test_apply_rerank_reorders_and_cuts,
        test_exclude_conf_skips_listed_folders,
        test_live_index_confirmed_miss_is_weak,
        test_z_and_coverage_best_are_independent_maxima,
    ]
    failures = skipped = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Skip as e:
            skipped += 1
            print(f"SKIP {t.__name__}: {e}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    if failures:
        raise SystemExit(f"{failures}/{len(tests)} tests failed")
    print(f"{len(tests) - skipped} passed, {skipped} skipped")
