# Prompt version history

| Version | Files | Status | Notes |
| --- | --- | --- | --- |
| v1 | `E_v1.txt`, `G_v1.txt` | frozen for evaluation | Written from the handoff contract (section 9) before any annotation. Same question / unchanged response / fallible reference for both conditions; same 25–60-word target. |

## Development review of v1 (30 discovery responses, phase `dev`)

- 60/60 outputs (E and G) passed schema and span validation on the first attempt; no corrective attempts were needed.
- Status mix (both conditions identical): 26 `supported_error`, 2 `insufficient_evidence` (final number with no shown computation), 1 `ambiguous` (question wording "cooked 9/10 kilograms" supports the response's literal reading; reference reads a fraction), 1 `no_error_observed` (reference appears to double-count gigs for "every other day for 2 weeks").
- Validation warnings: 2 descriptions coincidentally contain the gold number token (ratio expression "3"), 1 G evidence is only the final-answer line (abstention case).
- Recurring development patterns used to set reviewer granularity (see REVIEWER_GUIDE.md): direction reversal of comparisons; fraction applied to wrong base; omitted/double-counted quantity; execution error in a correctly written expression; correct-but-unevaluated expressions followed by an inconsistent final number; answering a different quantity.
- Decision: **no prompt revision**. The failure modes observed were content judgments (reference problems, unexplained numbers) that the v1 statuses already represent. No `dev_r2` phase was created.
- Caveat: the same agent session wrote E and G (`condition_context_isolated=false`).
