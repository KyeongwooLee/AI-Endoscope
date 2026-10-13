# Original-run exploratory diagnosis
Original test data are diagnostic only; no follow-up settings selected from these outcomes.

| Split | N | Keep | Full oracle | Repairable errors | Review changes |
|---|---:|---:|---:|---:|---:|
| pilot | 40 | 0.725 | 0.800 | 3 | 2 |
| train | 200 | 0.795 | 0.925 | 26 | 22 |
| validation | 80 | 0.738 | 0.863 | 10 | 8 |
| test | 100 | 0.650 | 0.790 | 14 | 11 |

## pilot
| Action | C→C | C→I | I→C | I→I | Unavailable | Unique repairs |
|---|---:|---:|---:|---:|---:|---:|
| Keep | 29 | 0 | 0 | 11 | 0 | 0 |
| Recalculate | 10 | 0 | 0 | 3 | 27 | 0 |
| Rebuild | 24 | 5 | 3 | 8 | 0 | 2 |
| Review | 29 | 0 | 1 | 10 | 0 | 0 |

```json
{
  "repair_overlap": {
    "Recalculate & Recalculate": 0,
    "Recalculate & Rebuild": 0,
    "Recalculate & Review": 0,
    "Rebuild & Recalculate": 0,
    "Rebuild & Rebuild": 3,
    "Rebuild & Review": 1,
    "Review & Recalculate": 0,
    "Review & Rebuild": 1,
    "Review & Review": 1
  },
  "calculator_reasons": {
    "no_explicit_final_numeric_equality": 27,
    "last_numeric_equality_with_rhs_matching_model_final_answer": 13
  },
  "review_changed_outcomes": {
    "1->1": 0,
    "1->0": 0,
    "0->1": 1,
    "0->0": 1
  }
}
```

## train
| Action | C→C | C→I | I→C | I→I | Unavailable | Unique repairs |
|---|---:|---:|---:|---:|---:|---:|
| Keep | 159 | 0 | 0 | 41 | 0 | 0 |
| Recalculate | 47 | 1 | 2 | 10 | 140 | 0 |
| Rebuild | 128 | 31 | 23 | 18 | 0 | 12 |
| Review | 154 | 5 | 12 | 29 | 0 | 3 |

```json
{
  "repair_overlap": {
    "Recalculate & Recalculate": 2,
    "Recalculate & Rebuild": 2,
    "Recalculate & Review": 0,
    "Rebuild & Recalculate": 2,
    "Rebuild & Rebuild": 23,
    "Rebuild & Review": 9,
    "Review & Recalculate": 0,
    "Review & Rebuild": 9,
    "Review & Review": 12
  },
  "calculator_reasons": {
    "last_numeric_equality_with_rhs_matching_model_final_answer": 60,
    "no_explicit_final_numeric_equality": 139,
    "no_final_answer": 1
  },
  "review_changed_outcomes": {
    "1->1": 0,
    "1->0": 5,
    "0->1": 12,
    "0->0": 5
  }
}
```

## validation
| Action | C→C | C→I | I→C | I→I | Unavailable | Unique repairs |
|---|---:|---:|---:|---:|---:|---:|
| Keep | 59 | 0 | 0 | 21 | 0 | 0 |
| Recalculate | 14 | 0 | 0 | 9 | 57 | 0 |
| Rebuild | 50 | 9 | 10 | 11 | 0 | 8 |
| Review | 57 | 2 | 2 | 19 | 0 | 0 |

```json
{
  "repair_overlap": {
    "Recalculate & Recalculate": 0,
    "Recalculate & Rebuild": 0,
    "Recalculate & Review": 0,
    "Rebuild & Recalculate": 0,
    "Rebuild & Rebuild": 10,
    "Rebuild & Review": 2,
    "Review & Recalculate": 0,
    "Review & Rebuild": 2,
    "Review & Review": 2
  },
  "calculator_reasons": {
    "no_explicit_final_numeric_equality": 57,
    "last_numeric_equality_with_rhs_matching_model_final_answer": 23
  },
  "review_changed_outcomes": {
    "1->1": 0,
    "1->0": 2,
    "0->1": 2,
    "0->0": 4
  }
}
```

```json
{
  "external": {
    "repair": {
      "n": 4,
      "mean_chosen_probability": 0.6331376965180637,
      "mean_keep_probability": 0.5545463468164622,
      "mean_margin": 0.07859134970160156
    },
    "harm": {
      "n": 0,
      "mean_chosen_probability": null,
      "mean_keep_probability": null,
      "mean_margin": null
    },
    "missed_repairable": {
      "n": 6,
      "mean_chosen_probability": 0.7569148644983453,
      "mean_keep_probability": 0.7177652195484036,
      "mean_margin": 0.039149644949941755
    },
    "all_interventions": {
      "n": 72,
      "mean_chosen_probability": 0.7564556243405389,
      "mean_keep_probability": 0.7212192748721813,
      "mean_margin": 0.035236349468357714
    }
  },
  "internal": {
    "repair": {
      "n": 5,
      "mean_chosen_probability": 0.7153340723568136,
      "mean_keep_probability": 0.5621297187478926,
      "mean_margin": 0.1532043536089209
    },
    "harm": {
      "n": 0,
      "mean_chosen_probability": null,
      "mean_keep_probability": null,
      "mean_margin": null
    },
    "missed_repairable": {
      "n": 5,
      "mean_chosen_probability": 0.8522299574931038,
      "mean_keep_probability": 0.7962119417894872,
      "mean_margin": 0.05601801570361673
    },
    "all_interventions": {
      "n": 58,
      "mean_chosen_probability": 0.8325028644499319,
      "mean_keep_probability": 0.761430261122174,
      "mean_margin": 0.07107260332775776
    }
  }
}
```

## test
| Action | C→C | C→I | I→C | I→I | Unavailable | Unique repairs |
|---|---:|---:|---:|---:|---:|---:|
| Keep | 65 | 0 | 0 | 35 | 0 | 0 |
| Recalculate | 22 | 1 | 0 | 12 | 65 | 0 |
| Rebuild | 51 | 14 | 12 | 23 | 0 | 10 |
| Review | 64 | 1 | 4 | 31 | 0 | 2 |

```json
{
  "repair_overlap": {
    "Recalculate & Recalculate": 0,
    "Recalculate & Rebuild": 0,
    "Recalculate & Review": 0,
    "Rebuild & Recalculate": 0,
    "Rebuild & Rebuild": 12,
    "Rebuild & Review": 2,
    "Review & Recalculate": 0,
    "Review & Rebuild": 2,
    "Review & Review": 4
  },
  "calculator_reasons": {
    "no_explicit_final_numeric_equality": 65,
    "last_numeric_equality_with_rhs_matching_model_final_answer": 35
  },
  "review_changed_outcomes": {
    "1->1": 0,
    "1->0": 1,
    "0->1": 4,
    "0->0": 6
  }
}
```

```json
{
  "external": {
    "repair": {
      "n": 2,
      "mean_chosen_probability": 0.6504972400164941,
      "mean_keep_probability": 0.6085914640488221,
      "mean_margin": 0.04190577596767209
    },
    "harm": {
      "n": 3,
      "mean_chosen_probability": 0.611313521870782,
      "mean_keep_probability": 0.5598123700555883,
      "mean_margin": 0.0515011518151938
    },
    "missed_repairable": {
      "n": 12,
      "mean_chosen_probability": 0.720103395651078,
      "mean_keep_probability": 0.6608468340637733,
      "mean_margin": 0.0592565615873046
    },
    "all_interventions": {
      "n": 91,
      "mean_chosen_probability": 0.7178256020878226,
      "mean_keep_probability": 0.6677217371862106,
      "mean_margin": 0.050103864901612075
    }
  },
  "internal": {
    "repair": {
      "n": 1,
      "mean_chosen_probability": 0.808200763192121,
      "mean_keep_probability": 0.586035266976466,
      "mean_margin": 0.22216549621565496
    },
    "harm": {
      "n": 5,
      "mean_chosen_probability": 0.7033513052831064,
      "mean_keep_probability": 0.6627410753313502,
      "mean_margin": 0.04061022995175623
    },
    "missed_repairable": {
      "n": 13,
      "mean_chosen_probability": 0.7609389474088938,
      "mean_keep_probability": 0.6646280940015449,
      "mean_margin": 0.09631085340734888
    },
    "all_interventions": {
      "n": 83,
      "mean_chosen_probability": 0.790107231520172,
      "mean_keep_probability": 0.7018144364738252,
      "mean_margin": 0.08829279504634681
    }
  }
}
```
