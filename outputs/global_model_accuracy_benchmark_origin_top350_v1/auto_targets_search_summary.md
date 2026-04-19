# Auto Target Ladder Search Summary

## Inputs
- predictions_csv: `outputs/global_model_accuracy_benchmark_origin_top350_v1/random_forest_raw_test_predictions.csv`
- samples: `80640`
- positive_rate: `0.056399`
- min_count: `10`
- step: `5.000%`

## Candidates

| name | monotonic | target_bins_weighted_abs_gap | cumulative_weighted_abs_gap | target_bins_count | targets |
|---|---:|---:|---:|---:|---|
| center | 1 | 0.001026 | 0.050792 | 20 | `97.5,92.5,87.5,82.5,77.5,72.5,67.5,62.5,57.5,52.5,47.5,42.5,37.5,32.5,27.5,22.5,17.5,12.5,7.5,2.5` |
| edge | 1 | 0.000743 | 0.053596 | 19 | `95,90,85,80,75,70,65,60,55,50,45,40,35,30,25,20,15,10,5` |

## Selected

- name: `center`
- targets: `97.5,92.5,87.5,82.5,77.5,72.5,67.5,62.5,57.5,52.5,47.5,42.5,37.5,32.5,27.5,22.5,17.5,12.5,7.5,2.5`
- target_bins_weighted_abs_gap: `0.001026`
- cumulative_weighted_abs_gap: `0.050792`

## Next Command

```bash
python src/fit_confidence_target_intervals.py \
  --predictions-csv outputs/global_model_accuracy_benchmark_origin_top350_v1/random_forest_raw_test_predictions.csv \
  --output-dir outputs/global_model_accuracy_benchmark_origin_top350_v1 \
  --min-count 10 \
  --targets "97.5,92.5,87.5,82.5,77.5,72.5,67.5,62.5,57.5,52.5,47.5,42.5,37.5,32.5,27.5,22.5,17.5,12.5,7.5,2.5"
```
