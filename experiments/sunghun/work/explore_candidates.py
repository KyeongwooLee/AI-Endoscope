from pathlib import Path
import pandas as pd
import numpy as np

root = Path('outputs/vessl-qwen-gsm8k-fp32-v6/analysis/exploration')
frame = pd.read_parquet(root / 'example_features.parquet')
frame = frame[frame.analysis_eligible]
features = ['entropy', 'margin', 'generated_logprob', 'js_final', 'adjacent_js',
            'top1_agreement', 'log_rank', 'support_change_abs', 'lower_tail_logprob',
            'stabilization_depth', 'late_stabilization_depth', 'top1_switches']
results = []
for (region, layer), group in frame.groupby(['region', 'layer']):
    for feature in features:
        valid = group.dropna(subset=[feature])
        y = ~valid.answer_correct
        a, b = valid.loc[~y, feature], valid.loc[y, feature]
        if not len(a) or not len(b):
            continue
        rank = valid[feature].rank(method='average')
        auc = (rank[y].sum() - len(b)*(len(b)+1)/2) / (len(a)*len(b))
        std = np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())/(len(a)+len(b)-2))
        results.append(dict(region=region, layer=int(layer), feature=feature,
                            correct_mean=a.mean(), incorrect_mean=b.mean(),
                            n_correct=len(a), n_incorrect=len(b),
                            error_auc=float(auc), oriented_auc=max(auc, 1-auc),
                            standardized_difference=(b.mean()-a.mean())/std if std else 0))
result = pd.DataFrame(results).sort_values('oriented_auc', ascending=False)
result.to_csv(root / 'exploratory_candidate_scan.csv', index=False)
print('Top candidates (training-set descriptive AUC, direction selected on train):')
print(result.head(14).to_string(index=False))
print('\nReasoning top candidates:')
print(result[result.region == 'reasoning'].head(8).to_string(index=False))
print('\nFinal layer confidence:')
print(result[(result.layer == 28) & result.feature.isin(['entropy','margin','generated_logprob'])].to_string(index=False))
group = frame[(frame.region == 'reasoning') & (frame.layer == 14)]
y = ~group.answer_correct
rank = group.generated_tokens.rank()
auc = (rank[y].sum() - y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum())
print('\nLength baseline oriented AUC:', max(auc, 1-auc),
      'means correct/incorrect:', group.loc[~y,'generated_tokens'].mean(), group.loc[y,'generated_tokens'].mean())
