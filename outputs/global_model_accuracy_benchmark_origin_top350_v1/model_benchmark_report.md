# Global Model Accuracy Benchmark

## Best By Recall-Aware F1 (Raw)

- model: `histgb_global`
- raw_bestf1_f1: `0.309674`
- raw_bestf1_precision: `0.248931`
- raw_bestf1_recall: `0.409631`
- raw_bestf1_threshold: `0.700833`
- raw_recall_at_p70: `0.000220`

## Test Set Comparison

| model | raw_auc | raw_pr_auc | raw_logloss | raw_bestf1_precision | raw_bestf1_recall | raw_bestf1_f1 | raw_recall_at_p70 | iso_logloss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| histgb_global | 0.813031 | 0.262559 | 0.449826 | 0.248931 | 0.409631 | 0.309674 | 0.000220 | 0.177182 |
| random_forest_global | 0.806610 | 0.254789 | 0.349148 | 0.244396 | 0.390721 | 0.300702 | 0.047274 | 0.178858 |
| xgboost_global | 0.789640 | 0.237622 | 0.364910 | 0.221296 | 0.397098 | 0.284208 | 0.043316 | 0.183186 |
