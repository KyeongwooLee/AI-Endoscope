# Feature definitions

Prediction input whitelist is `learn.inputs`. It excludes gold/reference answers, reference reasoning, original correctness, repair outputs, and observed action outcomes.

The standardized scalar columns, in order, are original mean raw causal token NLL (excluding tokenizer special IDs); log1p(non-special original response token count); generated-verdict CORRECT, INCORRECT and INVALID indicators; mean raw verdict token NLL; extracted numerical-answer token mean and max raw NLL; numerical-span availability; truncation; full CORRECT and INCORRECT candidate sequence log probabilities; the two-candidate normalized INCORRECT probability; and reference-free calculator availability. Missing numerical-token NLL uses zero with availability=false. The original confidence, all generated IDs, full per-token NLLs, extracted character spans, overlapping token indices, and decoded token pieces are retained.

D1 uses the first two scalar columns. D2 adds train-fit TF-IDF of question + original response (1,500 max unigram/bigram features, min_df2, sublinear term frequency). D3 adds the four old verifier columns. Strengthened external adds all remaining scalar columns. Each internal comparison uses exactly the strengthened external columns and TF-IDF, adding one hidden vector representation. Both variants see identical examples and missingness information.

Hidden vectors come from raw post-block14 outputs in the segmented verification prompt, before final model normalization. Positions: final token overlapping the extractor's numerical-answer character span; dedicated first response-separator token; final prompt token before verdict generation. An absent numerical span gives a zero vector and the shared availability indicator. These are reference-free v3 heuristic spans, not gold-aligned locations; documented prose-extraction ambiguities remain a limitation.

`scaled`: training-fitted coordinate StandardScaler divided by sqrt1536. `standard`: the same coordinate standardization without shrinkage. `pca32`: training-fitted randomized PCA32 (seed17, non-whitened), after training-fitted coordinate standardization. Shared scalars are individually standardized; TF-IDF retains its standard row L2 normalization. The relative hidden-block scale therefore interacts with the shared C regularization. No preprocessing fits on development or test.

Both-label scoring uses identical pre-verdict prompt IDs, then standalone CORRECT tokens [52320,44840] and INCORRECT tokens [38439,868,44840]. Sum autoregressive log probabilities over all label tokens, without EOS or length normalization. The normalized score conditions on these two candidate strings; it is not presumed calibrated correctness probability. No generated verdict token is present in any hidden-feature forward prefix.

Availability comes only from the unchanged calculator's reference-free extraction of the model's own original response. Four independent success heads train only on available labels. Neither an observed repair result nor correctness-group membership is an inference input.

Two untuned standalone controls are also reported: the generated-verdict error indicator (invalid=.5), and the normalized two-candidate INCORRECT probability. Neither participates in hidden-representation selection.

V3 measurement amendment: explicit final-answer-is markers and next-nonempty-line answers take priority; prose numerical predicates disambiguate contextual counts. All previous v2 records and changed labels are archived. This amendment occurred before classifier fitting and before any fresh-test generation.
