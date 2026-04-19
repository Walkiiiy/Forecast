# Recall-Aware Reassessment (Precision + Recall)

## 评估标准调整

- 不再只看“预测为连接时有多准（precision）”。
- 同时看“真实连接里抓到了多少（recall）”。
- 阈值在校准集上自动选择（`best-F1 threshold`），再到测试集报告。

## 模型对比（固定配置）

| model | raw_auc | raw_pr_auc | bestF1_precision | bestF1_recall | bestF1_F1 | iso_logloss |
|---|---:|---:|---:|---:|---:|---:|
| random_forest_global | 0.777424 | 0.207745 | 0.199303 | 0.285561 | 0.234760 | 0.173005 |
| lstm_global | 0.771395 | 0.169172 | 0.163258 | 0.315865 | 0.215258 | 0.176732 |
| histgb_global | 0.761246 | 0.184428 | 0.158690 | 0.314439 | 0.210929 | 0.177155 |
| xgboost_global | 0.736370 | 0.161580 | 0.139722 | 0.304100 | 0.191470 | 0.181306 |

按 best-F1 排名：
1. `random_forest_global` (F1=0.2348, Recall=0.2856, Precision=0.1993)
2. `lstm_global` (F1=0.2153, Recall=0.3159, Precision=0.1633)
3. `histgb_global` (F1=0.2109, Recall=0.3144, Precision=0.1587)
4. `xgboost_global` (F1=0.1915, Recall=0.3041, Precision=0.1397)

按 Recall 排名：
1. `lstm_global` (Recall=0.3159, Precision=0.1633, F1=0.2153)
2. `histgb_global` (Recall=0.3144, Precision=0.1587, F1=0.2109)
3. `xgboost_global` (Recall=0.3041, Precision=0.1397, F1=0.1915)
4. `random_forest_global` (Recall=0.2856, Precision=0.1993, F1=0.2348)

## LSTM 调参（以 Recall+F1 为目标）

结果文件：`outputs/lstm_global_tuning_recall_v1/lstm_tuning_results.csv`

| tag | auc | bestF1_precision | bestF1_recall | bestF1_F1 | recall_f1_score(0.6R+0.4F1) |
|---|---:|---:|---:|---:|---:|
| h128_l1_d01_lr1e3 | 0.706157 | 0.090207 | 0.570766 | 0.155792 | 0.404777 |
| h192_l2_d02_lr1e3 | 0.701011 | 0.090130 | 0.449911 | 0.150176 | 0.330017 |
| h128_l2_d03_lr5e4 | 0.710220 | 0.110791 | 0.370410 | 0.170566 | 0.290472 |
| h128_l2_d02_lr1e3 | 0.747280 | 0.127726 | 0.306952 | 0.180390 | 0.256327 |
| h64_l1_d01_lr1e3 | 0.764798 | 0.162819 | 0.238859 | 0.193642 | 0.220772 |

最佳高召回配置（当前实验）：
- `h128_l1_d01_lr1e3`
- Recall=0.5708, Precision=0.0902, F1=0.1558, AUC=0.7062

## 结论

- 如果你要“精确率与召回率平衡”作为主标准，当前第一仍是 `random_forest_global`（F1 最高且概率质量最好）。
- 如果你把“召回率”放在第一位，可以选高召回版 LSTM（Recall 可到约 0.57），但 precision 会降到约 0.09，误报会明显增加。
- 业务上建议先确定误报可接受范围，再在该范围内最大化召回。
