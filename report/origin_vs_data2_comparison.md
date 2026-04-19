# origin.csv 与 data2.csv 数据差异分析

## 1. 数据范围与结构

- 旧数据：`data/legacy/origin.csv`
- 新数据：`data/data2.csv`
- 旧数据时间范围：`2024-10-12 07:30:36` 到 `2026-01-08 23:21:48`
- 新数据时间范围：`2026-01-01 01:06:12` 到 `2026-03-05 23:48:13`

| 指标 | origin.csv | data2.csv |
|---|---:|---:|
| 记录数 | 85330 | 89566 |
| 用户数 | 129 | 1514 |
| VIN 数 | 129 | 1514 |
| 每用户样本中位数 | 260.0 | 17.0 |
| 每用户样本 P90 | 1928.2 | 167.7 |
| `start_lon` 缺失率 | 37.80% | 46.46% |
| `end_lon` 缺失率 | 37.80% | 46.46% |

字段命名差异：
- 旧数据经纬度字段是 `lon1/lat1/lon2/lat2`，新数据是 `start_lon/start_lat/end_lon/end_lat`。
- 核心键字段（`user_id/vin/start_time/end_time/connect_duration`）两者都存在。

## 2. 用户与 VIN 交集

- 用户交集数：`54`（旧 `129`，新 `1514`，Jaccard=`0.0340`）
- VIN 交集数：`54`（旧 `129`，新 `1514`，Jaccard=`0.0340`）
- 仅旧数据用户数：`75`；仅新数据用户数：`1460`

结论：新旧数据的主体用户群差异非常大，交集用户占比很低。

## 3. 时间窗口与事件交集

- 两数据时间交集窗口：`2026-01-01 01:06:12` 到 `2026-01-08 23:21:48`
- 该窗口内记录数：旧 `2986`，新 `875`
- 精确事件交集（`user_id+vin+start_time`）：`872` 条
- 交集事件占比：相对旧 `1.0234%`，相对新 `0.9737%`

结论：即使存在时间重叠，新旧数据的事件层重合比例也很低，说明新数据并不是旧数据的简单续增。

## 4. 同一用户在新旧数据上的差异

- 交集用户数：`54`
- 交集用户中有精确时间重合事件的用户数：`38`
- 交集用户中 `new_event_count > old_event_count`：`3`
- 交集用户中 `new_event_count < old_event_count`：`51`
- 交集用户 `new/old` 样本比中位数：`0.0887`

绝对变化最大的 Top 20 用户已输出到：
- `outputs/origin_vs_data2_comparison/shared_user_top_abs_delta.csv`

## 5. 可直接引用的结论

1. 两数据集不是同一批用户的平滑延伸，而是“少量交集 + 大量新增用户”的结构变化。
2. 交集用户中，大多数用户在新数据中的样本量明显少于旧数据。
3. 事件级交集占比低于 2%，说明不能直接把新数据当作旧数据的增量续采。
4. 若做模型对比或迁移评估，应优先在“交集用户 + 重叠时间窗口”下单独评测，避免样本分布漂移导致结论失真。

## 6. 产出文件
- `outputs/origin_vs_data2_comparison/dataset_summary.csv`
- `outputs/origin_vs_data2_comparison/overlap_summary.csv`
- `outputs/origin_vs_data2_comparison/shared_user_comparison.csv`
- `outputs/origin_vs_data2_comparison/shared_user_top_abs_delta.csv`
- `outputs/origin_vs_data2_comparison/only_old_users.csv`
- `outputs/origin_vs_data2_comparison/only_new_users.csv`