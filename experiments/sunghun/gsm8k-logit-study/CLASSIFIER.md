# Exploratory Error Classifiers

This local-only follow-up does not modify the original inference, parser, feature, or analysis modules. It uses already-extracted problem-level features. No VESSL/GPU/model-weight training is needed. Install the existing CPU dependencies and requirements-classifier.txt in a separate local environment.

Run from the project directory:

```powershell
python scripts/train_error_classifier.py --run ABSOLUTE_V6_RUN_PATH --output ABSOLUTE_NEW_OUTPUT_PATH
python -m unittest discover -s tests -p test_error_classifier.py
```

The output path must not exist. Original data is read only. Split IDs, protocol, preprocessing parameters, trained logistic models/MLP weights, validation selections, test predictions, paired bootstrap comparisons and report checksums are saved. Load pickle/joblib artifacts only from trusted sources.

The initial 800/200 exploration split is stratified by answer correctness using seed 42. Applying the pre-existing eligibility flag leaves 790 training and 198 validation problems; the test contains 1,294 eligible problems. Excluded answers and coverage remain documented in the original study reports. One eligible train and one eligible test problem lack reasoning features; these are imputed using only training-set medians plus missing indicators.

Incorrect is the positive class. Input groups: log response length; layer-28 confidence statistics; layers 7/14/21/27 internal statistics plus stabilization diagnostics; confidence plus internal; confidence plus length; all plus length. Internal JS/agreement diagnostics reference the final-layer prediction, so "internal" does not mean information measured without observing output distributions. Features are available after generation, not before solving the question. No gold values, IDs or labels are predictors.

Logistic C is selected from 0.01/0.1/1. MLP uses one 32-unit ReLU hidden layer, dropout 0.2, AdamW weight decay 0.01, learning rate 0.001/0.003, up to 300 full-batch epochs and validation-AUROC early stopping with patience 30. Predictions average seeds 42/43/44. Weighted training addresses class imbalance, but scores are not calibrated error probabilities. Standardization, missing-value handling, epoch/hyperparameter selection and max-F1 decision thresholds use only train/validation. Every selection is saved before test feature values are loaded.

This is an exploratory follow-up: the test's original aggregate accuracy and primary feature analysis had already been seen. Model selection is not based on test metrics. Paired problem-level bootstrap intervals condition on the fixed trained models and outcome counts; they do not include training variability and have no multiple-comparison correction.

Initial results: confidence+length logistic test AUROC 0.795 and AP 0.627. Adding internal features gives AUROC 0.795 and AP 0.634; neither paired improvement is established by the 95% bootstrap intervals. Without length, combined models improve AP over confidence-only models in exploratory intervals, but AUROC intervals still include zero. Thus the saved features have predictive signal, while incremental internal information beyond confidence plus response length is not established in this setup. MLP is not consistently better than logistic regression.
