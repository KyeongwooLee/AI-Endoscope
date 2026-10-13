# Minseon — GSM8K AI Endoscope A/B

> Publication status: code, reports, figures and summary results are ready for upload. Raw response/feature/model bundles are prepared locally but are NOT uploaded; public export requires additional authorization after automatic approval review blocked it. Data restoration instructions below apply once those bundles are published.

Frozen **Qwen2.5-1.5B-Instruct** experiments on error detection (A) and repair selection (B). Only small detectors/selectors are trained. **The stronger follow-up did not establish a hidden-state benefit.**

요약: GSM8K에서 hidden state를 추가했을 때 오류 탐지·수정 선택의 개선은 확립되지 않았다. 원본 실험과 후속 실험의 코드, 결과, 데이터 및 재현 절차를 보존했다.

## Start here

1. [Follow-up report](project/runs/20261012_followup/REPORT.md): primary results, paired uncertainty, learning curves, recovery/damage and cost.
2. [Original A/B report](project/runs/20261012_AB/REPORT.md) ([Korean](project/runs/20261012_AB/REPORT_KO.md)): the earlier, smaller experiment.
3. [Original-run diagnosis](project/runs/20261012_followup/DIAGNOSTICS.md), [scoring audit](project/runs/20261012_followup/SCORING_AUDIT.md), [readable errors](project/runs/20261012_followup/ERROR_CASES.md).
4. [REPRODUCE.md](REPRODUCE.md): installation, CPU checks, frozen evaluation and data restoration.
5. [DATA_README.md](DATA_README.md): artifact contents, labels, split boundaries and limitations.

## Main follow-up results

40 pilot + 1,200 training-pool + 200 development + 300 fresh official-test questions. Nested training sizes 200/500/1,000 with two training samples; primary = 1,000, seed 101. All choices frozen before generating the fresh test responses.

| Method | Result |
|---|---|
| A external | AUROC 0.803 |
| A + hidden | AUROC 0.810; difference +0.006, paired 95% CI [−0.020, +0.033] |
| B Keep | 69.7% accuracy |
| B validation-selected fixed Review | 72.3%; 11 errors repaired, 3 correct answers damaged |
| B external selector | 71.3%; 16 repaired, 11 damaged |
| B internal selector | 71.3%; 17 repaired, 12 damaged |
| Full observed-outcome oracle | 84.0%; this is not achievable selector performance |

The selectors did not establish an advantage over error-triggered fixed repair. The primary conservative margin was zero, so it matched the ordinary selector. Removing hidden-feature shrinkage did not resolve the weak gains. Recovery/damage predictors with too few minority outcomes were deferred. C, FinQA, larger models and activation steering were not run.

## Organization

Following [Sunghun's branch](https://github.com/KyeongwooLee/AI-Endoscope/tree/sunghun), this contribution lives on a personal branch under `experiments/minseon/`. The original code layout is nested under `project/` so its relative path resolution remains intact.

- `project/experiments/ab/`, `project/experiments/followup/`: byte-preserved experiment code and settings.
- `project/runs/`: browseable reports, figures, tables, configurations and predictions.
- `project/results/`: earlier meeting report and its builder.
- `data/`: compressed, checksum-verified run bundles split into 2 MiB parts. Run `python restore_data.py` to reconstruct the complete published records, hidden features and trained small models.
- `publication_manifest.json`: every archive/part/member hash, copied source hashes and explicit exclusions.

Sunghun used GitHub Releases for large data. This upload uses versioned archive parts on the branch because the connected upload interface has no release-asset operation. Both original A/B and follow-up artifacts are included, along with earlier runs needed for provenance. No Qwen weights, model cache, virtual environment, credentials, or separate source Notion/team documents are distributed. Local originals were not edited for publication.

Historical reports retain the machine paths and historical verification counts from execution; publication verification is separately recorded. All source code is retained without retuning or new GPU inference.
