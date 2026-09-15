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


def test_verdict_absent_mass_below_gate_unaffected():
    # Low absent mass: behaves exactly as before.
    assert server._verdict(z_best=5.0, coverage_best=0.9, absent_mass=0.1) == "STRONG"
    assert server._verdict(z_best=0.2, coverage_best=0.1, absent_mass=0.1) == "WEAK"
    assert server._verdict(z_best=2.0, coverage_best=0.4, absent_mass=0.1) == "MODERATE"


def test_verdict_gate_boundary():
    assert server._verdict(z_best=5.0, coverage_best=0.9, absent_mass=0.5) == "WEAK"
    assert server._verdict(z_best=5.0, coverage_best=0.9, absent_mass=0.49) == "STRONG"


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
        test_verdict_absent_mass_below_gate_unaffected,
        test_verdict_gate_boundary,
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
