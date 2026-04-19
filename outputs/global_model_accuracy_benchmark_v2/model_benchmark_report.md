# Global Model Accuracy Benchmark

## Best By Recall-Aware F1 (Raw)

- model: `random_forest_global`
- raw_bestf1_f1: `0.234760`
- raw_bestf1_precision: `0.199303`
- raw_bestf1_recall: `0.285561`
- raw_bestf1_threshold: `0.543377`
- raw_recall_at_p70: `nan`

## Test Set Comparison

| model | raw_auc | raw_pr_auc | raw_logloss | raw_bestf1_precision | raw_bestf1_recall | raw_bestf1_f1 | raw_recall_at_p70 | iso_logloss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_forest_global | 0.777424 | 0.207745 | 0.357816 | 0.199303 | 0.285561 | 0.234760 | nan | 0.173005 |
| histgb_global | 0.761246 | 0.184428 | 0.451899 | 0.158690 | 0.314439 | 0.210929 | nan | 0.177155 |
| xgboost_global | 0.736370 | 0.161580 | 0.327472 | 0.139722 | 0.304100 | 0.191470 | nan | 0.181306 |
