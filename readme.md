
新数据集基准评估
```bash
python src/benchmark_global_models.py \
  --data-folder data/top350 \
  --output-dir outputs/global_model_accuracy_benchmark_v2 \
  --seed 42
```

在新数据集上做全局模型基准评估，输出模型对比表，确认主模型（通常是 RF）。

- 输入：`data/top350/*.csv`
- 核心脚本：`benchmark_global_models.py`
- 共享构建器：`confidence_forecast_extreme_rebuild.py` 中 `build_dataset`

怎么做（脚本解释）：
- 用 `BuildConfig` 构建 train/calib/test（默认 `history_days=21`、`min_days=24`、`train=0.7`、`calib=0.15`）
- 训练 `histgb_global`、`xgboost_global`、`random_forest_global`
- 在 calib 上找 best-F1 阈值，在 test 上报告 AUC/PR-AUC/logloss/F1 等
- 生成按 `raw_bestf1_f1` 排序的模型对比

中间和结果输出文件：
- `outputs/global_model_accuracy_benchmark_v2/model_benchmark_table.csv`
- `outputs/global_model_accuracy_benchmark_v2/model_benchmark_report.md`
- `outputs/global_model_accuracy_benchmark_v2/benchmark_summary.json`

---

新数据集 RF 5% 分箱导出
```bash
python src/export_random_forest_raw_confidence_5pct.py \
  --data-folder data/top350 \
  --output-dir outputs/global_model_accuracy_benchmark_v2 \
  --seed 42
```


训练 RF，并导出测试集原始概率在 5% 分箱下的统计结果和累计结果。


- 输入：`data/top350/*.csv`
- 脚本：`export_random_forest_raw_confidence_5pct.py`
- 共享构建器：`build_dataset`

怎么做（脚本解释）：
- 用同样的数据切分构建样本
- 训练 RF（`n_estimators=500`，`max_depth=24`，`min_samples_leaf=20`）
- calib 集上搜索 best-F1 阈值
- test 集导出：
  - 单桶 5% 分箱统计（`00-05` 到 `95-100`）
  - `>=阈值` 的累计统计
  - 每条测试样本的原始概率预测

中间和结果输出文件：
- 中间文件（后续会被使用）：
  - `outputs/global_model_accuracy_benchmark_v2/random_forest_raw_test_predictions.csv`
- 结果文件：
  - `outputs/global_model_accuracy_benchmark_v2/random_forest_raw_confidence_5pct.csv`
  - `outputs/global_model_accuracy_benchmark_v2/random_forest_raw_confidence_5pct_cumulative.csv`
  - `outputs/global_model_accuracy_benchmark_v2/random_forest_raw_confidence_5pct_summary.json`
  - `outputs/global_model_accuracy_benchmark_v2/random_forest_raw_confidence_5pct.md`
  - `outputs/global_model_accuracy_benchmark_v2/random_forest_raw_confidence_5pct_cumulative.md`

---

旧数据集基准评估
```bash
python src/benchmark_global_models.py \
  --data-folder data/legacy/top350_from_origin \
  --output-dir outputs/global_model_accuracy_benchmark_origin_top350_v1 \
  --seed 42
```


在旧数据集（origin top 用户子集）上做同样的模型基准评估。


- 输入：`data/legacy/top350_from_origin/*.csv`
- 脚本：`benchmark_global_models.py`
- 共享构建器：`build_dataset`


流程与“指令 1”一致，只是数据源切换为旧数据集用户文件夹。

中间和结果输出文件：
- `outputs/global_model_accuracy_benchmark_origin_top350_v1/model_benchmark_table.csv`
- `outputs/global_model_accuracy_benchmark_origin_top350_v1/model_benchmark_report.md`
- `outputs/global_model_accuracy_benchmark_origin_top350_v1/benchmark_summary.json`

---

旧数据集 RF 5% 分箱导出
```bash
python src/export_random_forest_raw_confidence_5pct.py \
  --data-folder data/legacy/top350_from_origin \
  --output-dir outputs/global_model_accuracy_benchmark_origin_top350_v1 \
  --seed 42
```


在旧数据集训练 RF 并导出后续区间拟合需要的原始预测文件与分箱统计。


- 输入：`data/legacy/top350_from_origin/*.csv`
- 脚本：`export_random_forest_raw_confidence_5pct.py`
- 共享构建器：`build_dataset`


与“指令 2”相同，输出旧数据集对应的 `random_forest_raw_*` 文件。

中间和结果输出文件：
- 中间文件（后续自动寻优和区间拟合要用）：
  - `outputs/global_model_accuracy_benchmark_origin_top350_v1/random_forest_raw_test_predictions.csv`
- 结果文件：
  - `outputs/global_model_accuracy_benchmark_origin_top350_v1/random_forest_raw_confidence_5pct.csv`
  - `outputs/global_model_accuracy_benchmark_origin_top350_v1/random_forest_raw_confidence_5pct_cumulative.csv`
  - `outputs/global_model_accuracy_benchmark_origin_top350_v1/random_forest_raw_confidence_5pct_summary.json`
  - `outputs/global_model_accuracy_benchmark_origin_top350_v1/random_forest_raw_confidence_5pct.md`
  - `outputs/global_model_accuracy_benchmark_origin_top350_v1/random_forest_raw_confidence_5pct_cumulative.md`

---

自动寻优 targets（独立脚本）
```bash
python src/auto_optimize_confidence_targets.py \
  --predictions-csv outputs/global_model_accuracy_benchmark_origin_top350_v1/random_forest_raw_test_predictions.csv \
  --output-dir outputs/global_model_accuracy_benchmark_origin_top350_v1 \
  --min-count 10 \
  --candidate-modes edge,center \
  --step 5
```

这条指令在做什么：
自动在候选 `targets` 序列中选最优，并给出可直接复用的 `targets` 字符串。

用什么：
- 输入：`random_forest_raw_test_predictions.csv`
- 脚本：`auto_optimize_confidence_targets.py`
- 内部复用：`fit_confidence_target_intervals.py` 的 `fit_intervals` 和 `make_cumulative_from_intervals`

怎么做（脚本解释）：
- 默认比较两组候选：
  - `edge`：`95,90,...,5`
  - `center`：`97.5,92.5,...,2.5`
- 对每组候选执行一次完整区间拟合并计算指标
- 选择规则（排序优先级）：
  1. 区间实际率是否单调非增
  2. `cumulative_weighted_abs_gap` 越小越优
  3. `target_bins_weighted_abs_gap` 越小越优

中间和结果输出文件：
- `outputs/global_model_accuracy_benchmark_origin_top350_v1/auto_targets_search_summary.json`
- `outputs/global_model_accuracy_benchmark_origin_top350_v1/auto_targets_search_summary.md`
- `outputs/global_model_accuracy_benchmark_origin_top350_v1/auto_selected_targets.txt`

---

使用自动选出的 targets 做区间拟合
```bash
TARGETS=$(cat outputs/global_model_accuracy_benchmark_origin_top350_v1/auto_selected_targets.txt)
python src/fit_confidence_target_intervals.py \
  --predictions-csv outputs/global_model_accuracy_benchmark_origin_top350_v1/random_forest_raw_test_predictions.csv \
  --output-dir outputs/global_model_accuracy_benchmark_origin_top350_v1 \
  --targets "$TARGETS" \
  --min-count 10
```

这条指令在做什么：
按自动选出的 `targets` 生成旧数据集最终的动态区间拟合结果（含累计表）。

用什么：
- 输入：
  - `random_forest_raw_test_predictions.csv`
  - `auto_selected_targets.txt`
- 脚本：`fit_confidence_target_intervals.py`

怎么做（脚本解释）：
- 按 `raw_confidence` 从高到低排序样本
- 对每个目标率在满足 `min_count` 的端点范围内搜索最优切分点
- 生成单区间表 `confidence_target_intervals.csv`
- 再生成累计区间表 `confidence_target_intervals_cumulative.csv`

中间和结果输出文件：
- 结果文件：
  - `outputs/global_model_accuracy_benchmark_origin_top350_v1/confidence_target_intervals.csv`
  - `outputs/global_model_accuracy_benchmark_origin_top350_v1/confidence_target_intervals_cumulative.csv`
  - `outputs/global_model_accuracy_benchmark_origin_top350_v1/confidence_target_intervals_summary.json`
  - `outputs/global_model_accuracy_benchmark_origin_top350_v1/confidence_target_intervals_report.md`


如果需要固定到历史同一组 `targets`，可在第 6 步直接写死：
`97.5,92.5,87.5,82.5,77.5,72.5,67.5,62.5,57.5,52.5,47.5,42.5,37.5,32.5,27.5,22.5,17.5,12.5,7.5,2.5`。
