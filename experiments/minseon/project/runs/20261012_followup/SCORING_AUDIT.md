# Scoring audit (exploratory original data)

A seeded stratified sample of 36 responses was read: 12 harmful revisions, 12 numerical-versus-strict disagreements, and 12 random agreements. Full sampling frame: scoring_audit_cases.json (514 rows). Individual reviewed decisions: manual_adjudications.json. This is an assistant qualitative audit, not independent blinded human annotation.

Two sampled apparent harmful revisions were actually extraction failures: train_16 Review says 800 before mentioning a 200 shirt; test_430 Rebuild says 132 before mentioning a 20% discount. Twelve sampled numerical/strict disagreements were formatting differences rather than numerical errors. A nonnumeric refusal borrowed a reasoning number, without changing its failure label. The known test_823 reference error remains separately flagged in the original run.

Version 2 is frozen before follow-up collection: preserve explicit numeric markers and boxes; for prose final-answer lines extract the first number on that line (or mark unavailable if no number); otherwise take the earliest concluding numerical clause, extending have/has/will have; then retain the prior fallback. These are reference-free measurement rules, motivated by original training examples 16, 7227, 7188, 7120, and 6293. Old test cases are exploratory checks only, never used to choose model settings. No old record or label is changed. All follow-up records retain legacy and strict scores for sensitivity.

All 15 changed old-run extractions are in scoring_v2_changes_on_old.json. V2 repairs seven correctness labels (five training and two exploratory test); no correct label becomes wrong. It remains imperfect: train_66 Rebuild's prose contains a contextual 4 before its claimed -15, and train_5009 gives a symbolic expression. Both remain incorrect; these cases are explicitly retained rather than concealed. Rule-based numerical scoring does not certify reasoning, resolve all ambiguities, or constitute comprehensive reference verification.

The conservative calculator is unchanged, including its original answer extraction for eligibility. Thus calculator availability is not silently expanded by scorer v2; its returned number is scored consistently against the same gold as every other action.

## Pre-fit v3 amendment

A training-only audit during collection found regressions in v2: a final answer placed on the next nonempty line, a `final answer is 30` marker after another quantity in the paragraph, and a contextual count preceding the actual result (`The 2 boxes contain a total of 2304 soaps`). Collection was paused before classifier fitting and before any fresh-test generation.

V3 recognizes `final answer is` with or without a colon, follows a marker onto the next nonempty line, and prefers an explicit numerical predicate within a prose final-answer line. It also handles signed currency and marks unresolved symbolic final expressions unavailable. These are measurement corrections, not settings selected to favor any detector or selector.

The 312 completed records (40 pilot +272 training) are archived under measurement_v2_archive. Four action-success labels changed from incorrect to correct; all are training repairs. One original numerical-token span required re-extraction, with its old feature file preserved. All original/repair generations are reused. The exact ledger is scoring_v3_amendment.json. Scoring_v2_changes_on_old.json and original_v2_sensitivity.json remain explicitly historical v2 analyses. The final follow-up uses v3 consistently, with legacy and strict scores retained; residual heuristic extraction limitations remain.
