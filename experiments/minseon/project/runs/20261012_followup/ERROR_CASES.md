# Qualitative original-test repair cases
Exploratory, assistant-coded descriptions; overlapping error types, not validated error-class labels.

| ID | Supported error description | Successful actions | External choice | Internal choice |
|---|---|---|---|---|
| gsm8k_test_1278 | Arithmetic: upstream addition error; final subtraction faithfully executes wrong intermediate. | Rebuild | Review | Review |
| gsm8k_test_477 | Arithmetic: fails to subtract overlap despite writing correct inclusion-exclusion formula. | Rebuild | Review | Keep |
| gsm8k_test_41 | Relationship/target: misreads multiplier and fails to subtract flame reach. | Rebuild | Review | Review |
| gsm8k_test_257 | Unit conversion: per ten feet treated as per foot. | Rebuild | Review | Recalculate |
| gsm8k_test_996 | Algebra and target: arithmetic slip, then reports Helene instead of Wilfred. | Review | Review | Keep |
| gsm8k_test_1171 | Omitted multiplicity: only one of four bedrooms included. | Rebuild | Review | Recalculate |
| gsm8k_test_729 | Fraction semantics: voted and remaining fractions reversed. | Rebuild, Review | Review | Rebuild |
| gsm8k_test_721 | Quantity accounting: omits one foot and misallocates duration. | Rebuild | Review | Review |
| gsm8k_test_346 | Wrong operation: invents weekly division instead of counting lunches. | Review | Rebuild | Rebuild |
| gsm8k_test_1159 | Conditional denominator and percent-unit error. | Rebuild | Keep | Review |
| gsm8k_test_1205 | Wrong multiplicity for mean: three dogs used rather than four. | Rebuild | Keep | Keep |
| gsm8k_test_517 | Ambiguous language: interprets two times more as three times; official convention uses twice. | Rebuild | Review | Review |
| gsm8k_test_715 | Unit/total confusion: per-computer budget treated as total. | Rebuild, Review | Recalculate | Recalculate |
| gsm8k_test_728 | Arithmetic: 15+45 represented as 36. | Rebuild | Review | Review |

Rebuild supplies most unique successes, often by restoring missing quantities or relations. Review also repairs some relational mistakes, so its opportunity is not confined to arithmetic. The unchanged calculator can execute only a final explicit equality; it cannot correct an earlier incorrect intermediate or a wrong relation. These mechanisms are supported by the examples, not by causal feature interventions.
