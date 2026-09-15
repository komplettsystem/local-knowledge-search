# Follow-up: the verdict still calls a confirmed miss MODERATE

> Example queries and file names from the private test corpus are anonymized in this public copy. All measurements (z, coverage, absent mass, ranks) come from the original queries against that corpus.

**Date:** 2026-09-13
**Repo:** `local-knowledge-search`
**Files:** `server.py` (verdict logic), `lib.py` (IDF)

The signal work from the first handover landed and is good. All five signals are
computed, nothing is filtered, `BM25.idf_for_term` handles the df=0 case correctly
rather than falling back to 0.0. This is a follow-up on the verdict layer only.

## Measured, on the live index

| signal | good query | Valencia (confirmed miss) |
|---|---|---|
| rare_terms (top hit) | 3/3 | **0/3** |
| raw cos | 0.818 | 0.605 |
| bm25 | 23.0 | 4.9 |
| within-set cos σ | 0.0232 | **0.0044** |
| z_best | +3.9σ | +2.4σ |
| coverage_best | 81% | 42% |
| **verdict** | STRONG | **MODERATE** |

Queries used:
- `team workspace memory architecture context and state` (hit)
- `Valencia apartment purchase Airbnb` (miss — corpus contains nothing on Valencia)

## Root cause

`_verdict()` takes `z_best` and `coverage_best`. For the miss: z 2.4 clears
`Z_WEAK=1.0`, coverage 0.42 clears `COVERAGE_WEAK=0.25`, so neither WEAK condition
fires and it falls through to MODERATE.

The 42% is the problem. Coverage is IDF-mass-weighted:

```python
coverage = sum(term_idf[t] for t in matched_terms) / total_idf_mass
```

For this query, `valencia` and `airbnb` both have df=0 and carry the bulk of the
IDF mass. `apartment` and `purchase` matched and are common. So 42% means
"matched the throwaway terms, missed both identifying ones" — which is the
signature of a miss, reported as a middling score.

**The decisive fact is already computed and then discarded.** `idf_for_term`
knows these terms appear in zero documents. That is not a weak signal to be
averaged against others; it is close to proof. Right now it gets laundered
through a weighted mean where in-corpus terms outvote it.

## Changes

### 1. Short-circuit on absent query terms (the actual fix)

Compute the share of query IDF mass held by terms with df=0:

```python
absent_terms = [t for t in distinct_terms if t not in bm25.idf]
absent_mass = sum(term_idf[t] for t in absent_terms) / total_idf_mass
```

For the Valencia query this is ~0.7. Gate on it in `_verdict()`: above ~0.5,
return WEAK regardless of z and coverage. Nothing the corpus does elsewhere can
compensate for the identifying term existing in zero documents.

Surface it in the header too — `2 query terms absent from corpus: [valencia,
airbnb]` says more to a reader than any percentage.

### 2. Stop taking independent maxima

`z_best` and `coverage_best` are each `max()` over the returned set, so they can
come from different chunks — an optimistic bound on a hypothetical result that
does not exist. Use the top-ranked chunk's values, or require both maxima to come
from the same chunk.

### 3. Feed dispersion into the verdict

Within-set cosine σ separated 5x on these two queries (0.0044 vs 0.0232) — better
than z did. It is computed and printed but never reaches `_verdict()`. A flat
returned set means the ranking found nothing to discriminate on, which is exactly
the miss case. Wire it in.

### 4. Note on Z_STRONG

The comment says 3.0 was picked against a measured miss at 2.4 and hits at ≥3.55.
That is a ~0.6σ margin fitted on two observations. It held here, but z is the
weakest of the five discriminators (1.6x separation vs 5x for dispersion and a
clean binary for rare_terms). Leave the value, demote its weight.

## Acceptance

- `Valencia apartment purchase Airbnb` → **WEAK**, header names `valencia` and
  `airbnb` as absent from the corpus, all 3 results still returned with full text.
- `team workspace memory architecture context and state` → still STRONG.
- A semantic-only case: correct hit sharing no rare literal term with the query.
  Confirm the df=0 gate does **not** fire (its terms exist in the corpus, just not
  in that chunk) and z carries the verdict. If the gate misfires here, it is too
  aggressive and should key on corpus-absence only, never per-chunk absence.

The distinction in that last case is the one to get right: **a term absent from
the whole corpus is decisive; a term absent from one chunk is not.**

## Status (2026-09-13)

**Implemented: #1 (absent-term gate).**
`server.py`: `_verdict()` now takes `absent_mass` and returns `WEAK`
unconditionally when it's `>= ABSENT_MASS_WEAK = 0.5`, before the existing
z/coverage checks run. `absent_terms`/`absent_mass` are computed once in
`_search()` from `bm25.idf` membership (corpus-wide by construction, so the
whole-corpus-vs-per-chunk distinction in the Acceptance section is satisfied
structurally, not by a threshold choice). Both are surfaced in the header:
`"N query term(s) absent from the entire corpus: [...] (NN% of query IDF
mass)"`. Measured (re-run under the current code): Valencia query →
`absent_mass = 0.577`, `z_best = 2.36`, `coverage_best = 0.423` → `WEAK`;
known-good query → `absent_mass = 0.0` → unaffected, still `STRONG`
(`z_best = 3.9`, `coverage_best = 0.808`). (The 42%/2.4σ figures in the
original table above and in the README are `best-in-set`, i.e. `max()` over
the returned set — see below; they were not changed.)

**Tried and reverted: #2 (single-chunk signals).**
First attempt: restricted `z_best`/`coverage_best` to `results[0]` (the
top-ranked, RRF-fused chunk) instead of independent maxima, on the theory
that (a) independent maxima can describe a `z` and a `coverage` that never
co-occurred in one chunk, and (b) a spuriously high `z` on some chunk RRF
didn't even rank first could manufacture a false `STRONG`. Checked this
against the acceptance tests, then stress-tested it against 25 real queries
against the live index comparing old (max-over-set) vs. new (top-ranked)
verdicts. Result: 3/25 verdicts drifted, all `STRONG → MODERATE`, zero in
the other direction, and the drifts were **real hits being downgraded**, not
false positives being corrected. Concrete case: query "dispute a parking
ticket" — the actually-matching doc
(`admin/procedures/transport/parking-fine-appeal.md`)
ranked **6th** by RRF fusion (rank 1 was an unrelated PM-workflow doc, low
literal/vector overlap on this query pushed it down) but carried the
highest z in the returned set (3.13, vs. rank 1's 2.45). Restricting to
rank 1 lost that signal and reported `MODERATE` for what both old code and
manual inspection agree is a genuine `STRONG` hit. The theorized failure
mode in (b) never showed up in any of the 25 queries. Net: (a) is a real but
minor interpretability nit not worth (b)'s unproven benefit and this
measured recall cost. Reverted to independent `max()` over the returned
set, matching the original code and its original comment's rationale.
`test_relevance.py`'s `test_z_and_coverage_best_are_independent_maxima`
pins this down using the parking-ticket case, so a future attempt at this
doesn't silently reintroduce the regression without re-deriving why it was
reverted.

If interpretability (not recall) is the actual goal here, the arguably
better fix is naming which chunk each of `z_best`/`coverage_best` came from
in the header, rather than constraining the metric itself — not done here
since it wasn't asked for and the original ask was about the miss/hit
verdict split, not the header format.

**Deferred: #3 (wire in dispersion σ) and #4 (demote Z_STRONG's weight).**
Not implemented. Reasoning, surfaced before doing any work rather than
after:
- #3's proposed σ threshold would be calibrated on the same two queries
  this doc measures — exactly the "fitted on two observations" problem
  this doc calls out for `Z_STRONG` (see "Note on Z_STRONG" above). Also a
  specific failure mode isn't ruled out yet: within-set σ is measured over
  the *returned* top-k, so a genuine hit in a corpus with several
  near-duplicate/overlapping docs on the same topic could show low σ
  despite being correct — conflating "top match's novelty vs. its
  neighbors" with "relevance." Needs more query/corpus combinations,
  especially topically-dense ones, before it gates anything.
- #4 as stated ("demote its weight") isn't directly actionable: `_verdict()`
  is a hard gate ladder (if/elif), not a weighted score, so there's no
  weight to demote without first deciding whether the verdict becomes a
  scored combination of signals — an architecture change, not a threshold
  edit. Left `Z_STRONG` untouched pending that decision.

**New finding from testing the Acceptance section's semantic-only case:**
query `"categorizing memory kinds for people and AI agents collaborating"`
against the known `workspace-memory-architecture.md` chunk 1 (no
literal overlap on `workspace`, `state`, `context`, `episodic`)
correctly returned `STRONG` (z=4.53) and the gate correctly did not fire —
confirming the corpus-wide-vs-per-chunk distinction holds in practice, not
just by construction. But it was closer than expected: `categorizing` and
`collaborating` are themselves absent from the *whole* corpus (paraphrase
drift, not identifying vocabulary), pushing `absent_mass` to 0.462 — just
under the 0.5 gate. A paraphrase using one more such word on a shorter query
could cross the gate and wrongly force `WEAK` on a genuine semantic hit.
Worth watching; not worth lowering `ABSENT_MASS_WEAK` preemptively on one
data point (same objection as #3/#4), but if this recurs, the fix is likely
scoping the gate to distinctive/rare terms (e.g. only terms that would also
qualify for `rare_terms`) rather than the full absent-term set, so ordinary
paraphrase vocabulary can't accumulate mass on its own.

Tests: `test_relevance.py` (new, plain-assertion, no framework) — verdict-gate
unit tests, a live-index integration test (confirmed miss → `WEAK` with
`valencia`/`airbnb` named; known-good query → `STRONG`), and the
independent-maxima regression test for the reverted #2 attempt.

## Response to the implementation findings (2026-09-13)

Conceding #2, accepting #3 and #4, and adding two things that fall out of the
test data.

### #2: the revert was right and the original recommendation was wrong

The parking-ticket case settles it. If RRF ranked the correct document 6th while
its z ranked highest in the set, then constraining the signals to `results[0]`
inherits the exact defect this whole workstream exists to route around — RRF's
ranking being unreliable. Recommending it was a mistake: it would have made the
judgement layer depend on the ranking layer it was built to compensate for.
3/25 drifting one-way with zero corrections is decisive, and pinning it in
`test_relevance.py` is the right call.

The interpretability nit is real but minor, and naming the source chunk in the
header is the better fix. Agreed it wasn't the ask.

### #2 has a corollary neither of us drew

If z ranked the correct document above RRF on that query, the signals added for
*judging* may be better at *ranking* than the fused score is. Worth measuring on
the 25-query set that already exists: compare the rank of the human-judged
correct document under RRF vs under z vs under a simple combination. If z wins
often, the fix is bigger than a verdict line — the fusion step itself should be
reconsidered, and the "no threshold can work on a rank-only score" argument from
the first handover becomes an argument about ranking, not just reporting.

Measurement first. One query is an anecdote.

### #3 and #4: accepted as reasoned

The near-duplicate objection to #3 is one I missed and it is specific to this
corpus — a folder of technology notes is full of topically adjacent documents, exactly
the shape that would show low within-set σ on a genuine hit. Deferring pending
topically-dense test cases is correct.

#4 was sloppy on my part. "Demote its weight" presumes a weighted score; the
verdict is a gate ladder and has no weights. Leaving `Z_STRONG` alone pending a
decision on whether the verdict becomes scored is the right reading.

### The new finding is the most important thing here

`absent_mass = 0.462` against a `0.5` gate, on a genuine hit, is a 0.04 margin
holding up a correct answer. The proposed fix (scope the gate to terms that
would qualify for `rare_terms`) points the right way but does not go far enough,
because of a flaw in the original design of the gate:

**IDF cannot discriminate among absent terms.** `idf_for_term` returns the same
maximum value for every df=0 term, so `valencia` and `categorizing` are
indistinguishable to it. One is the subject of the query; the other is paraphrase
vocabulary. Restricting to the top-N rarest does not separate them — they tie,
and the tiebreak is alphabetical.

So the gate needs a signal that is not frequency-based. Three candidates, cheapest
first:

1. **Capitalization in the raw query.** `tokenize()` lowercases before anything
   else sees the string. `Valencia` and `Airbnb` are capitalized mid-sentence;
   `categorizing` and `collaborating` are not. Keeping the pre-lowercase form
   alongside each token and requiring absent terms to be capitalized before they
   count toward `absent_mass` would separate the two measured cases cleanly, at
   the cost of one extra list. Fails on lowercase-typed proper nouns, which is a
   real but recoverable miss (it makes the gate less aggressive, never more).

2. **Embedding proximity to the corpus vocabulary.** An absent term close to
   in-corpus terms is paraphrase drift (`categorizing` sits near `categories`,
   `kinds`); one far from everything is a genuinely new entity (`Valencia` sits
   near nothing in these notes). More robust than capitalization, costs one
   embedding per absent term plus a nearest-neighbour lookup.

3. **Out-of-vocabulary check against a common-English word list.** Crude, no
   model, and `Airbnb` may well be in such a list. Mentioned for completeness;
   probably not worth it.

(1) is the pragmatic choice and (2) is the correct one. Given that this corpus
identifies things overwhelmingly by proper noun (place names, tool names, project
names), (1) likely captures most of the value.

Either way the framing to keep: the gate should fire on **absent identifying
terms**, not absent terms. The first handover collapsed those two and this
finding is what exposed it.
