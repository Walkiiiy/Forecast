```python?code_reference&code_event_index=2
import pandas as pd
import numpy as np

# Table 1: Unconnected
counts_t1 = [12171, 3977, 2813, 2721] # derived from Table 3
preds_t1 = [0.980608, 0.927148, 0.875846, 0.824094]
acts_t1 = [0.996960, 0.991199, 0.988624, 0.973907]
logs_t1 = [0.031007, 0.098352, 0.154626, 0.233538]

cum_counts_t1 = []
cum_preds_t1 = []
cum_acts_t1 = []
cum_logs_t1 = []

curr_count = 0
curr_pred_sum = 0
curr_act_sum = 0
curr_log_sum = 0

labels_t1 = ["<= 05%", "<= 10%", "<= 15%", "<= 20%"]

for i in range(4):
    c = counts_t1[i]
    curr_count += c
    curr_pred_sum += preds_t1[i] * c
    curr_act_sum += acts_t1[i] * c
    curr_log_sum += logs_t1[i] * c
    
    cum_counts_t1.append(curr_count)
    cum_preds_t1.append(curr_pred_sum / curr_count)
    cum_acts_t1.append(curr_act_sum / curr_count)
    cum_logs_t1.append(curr_log_sum / curr_count)

df_t1 = pd.DataFrame({
    '预测未连接置信度区间': labels_t1,
    '累计样本数': cum_counts_t1,
    '累计平均预测置信度': cum_preds_t1,
    '累计真实未连接发生率': cum_acts_t1,
    '差值': np.array(cum_preds_t1) - np.array(cum_acts_t1),
    '累计logloss': cum_logs_t1
})

print("Table 1 Cumulative (Unconnected):")
print(df_t1.to_markdown(index=False))


# Table 2: Connected (Accumulate from 100% down to 70%)
counts_t2 = [347, 179, 133, 95, 76, 17] # 70-75 to 95-100
preds_t2 = [0.722280, 0.772152, 0.825690, 0.872957, 0.921283, 0.959687]
acts_t2 = [0.253602, 0.346369, 0.518797, 0.610526, 0.697368, 0.882353]
logs_t2 = [1.039352, 1.052667, 0.938397, 0.877410, 0.830634, 0.401732]

# Reverse lists to accumulate from highest prob to lowest
counts_t2 = counts_t2[::-1]
preds_t2 = preds_t2[::-1]
acts_t2 = acts_t2[::-1]
logs_t2 = logs_t2[::-1]

cum_counts_t2 = []
cum_preds_t2 = []
cum_acts_t2 = []
cum_logs_t2 = []

curr_count = 0
curr_pred_sum = 0
curr_act_sum = 0
curr_log_sum = 0

labels_t2 = [">= 95%", ">= 90%", ">= 85%", ">= 80%", ">= 75%", ">= 70%"]

for i in range(6):
    c = counts_t2[i]
    curr_count += c
    curr_pred_sum += preds_t2[i] * c
    curr_act_sum += acts_t2[i] * c
    curr_log_sum += logs_t2[i] * c
    
    cum_counts_t2.append(curr_count)
    cum_preds_t2.append(curr_pred_sum / curr_count)
    cum_acts_t2.append(curr_act_sum / curr_count)
    cum_logs_t2.append(curr_log_sum / curr_count)

df_t2 = pd.DataFrame({
    '预测连接置信度区间': labels_t2,
    '累计样本数': cum_counts_t2,
    '累计平均预测置信度': cum_preds_t2,
    '累计真实连接发生率': cum_acts_t2,
    '差值': np.array(cum_preds_t2) - np.array(cum_acts_t2),
    '累计logloss': cum_logs_t2
})

print("\nTable 2 Cumulative (Connected):")
print(df_t2.to_markdown(index=False))


```
```text?code_stdout&code_event_index=2
Table 1 Cumulative (Unconnected):
| 预测未连接置信度区间   |   累计样本数 |   累计平均预测置信度 |   累计真实未连接发生率 |         差值 |   累计logloss |
|:-------------|--------:|------------:|-------------:|-----------:|------------:|
| <= 05%       |   12171 |    0.980608 |     0.99696  | -0.016352  |   0.031007  |
| <= 10%       |   16148 |    0.967442 |     0.995541 | -0.0280995 |   0.047593  |
| <= 15%       |   18961 |    0.953853 |     0.994515 | -0.0406622 |   0.0634721 |
| <= 20%       |   21682 |    0.937569 |     0.991929 | -0.0543601 |   0.0848147 |

Table 2 Cumulative (Connected):
| 预测连接置信度区间   |   累计样本数 |   累计平均预测置信度 |   累计真实连接发生率 |       差值 |   累计logloss |
|:------------|--------:|------------:|------------:|---------:|------------:|
| >= 95%      |      17 |    0.959687 |    0.882353 | 0.077334 |    0.401732 |
| >= 90%      |      93 |    0.928303 |    0.731182 | 0.197121 |    0.752233 |
| >= 85%      |     188 |    0.900336 |    0.670212 | 0.230123 |    0.815487 |
| >= 80%      |     321 |    0.869408 |    0.607476 | 0.261931 |    0.866412 |
| >= 75%      |     500 |    0.83459  |    0.514    | 0.32059  |    0.933092 |
| >= 70%      |     847 |    0.788579 |    0.40732  | 0.381259 |    0.976624 |

```
为了将这两个区间表转换为累计（Cumulative）形式，我结合您提供的第三张表（即全量累计表）中的 `cumulative_count` 反推了每个区间的实际样本权重，并基于样本权重为您计算了精确的加权累计平均值。

根据逻辑：
* **未连接表**通常是从最低置信度开始向上累计（即 `<= 5%`，`<= 10%`...）。
* **连接表**通常是从最高置信度开始向下累计（即 `>= 95%`，`>= 90%`...），这也完全对应了您提供的大表中的数值。

以下是转换后的累计形式表格：

### 1. 预测未连接置信度（从低到高累计）

| 预测未连接置信度 | 累计样本数 | 累计平均预测置信度 | 累计真实未连接发生率 | 差值 (预测-真实) | 累计 logloss |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **<= 05%** | 12,171 | 0.980608 | 0.996960 | -0.016352 | 0.031007 |
| **<= 10%** | 16,148 | 0.967442 | 0.995541 | -0.028100 | 0.047593 |
| **<= 15%** | 18,961 | 0.953853 | 0.994515 | -0.040662 | 0.063472 |
| **<= 20%** | 21,682 | 0.937569 | 0.991929 | -0.054360 | 0.084815 |

*(注：由于您之前的要求将这部分转换为了 $1-x$，此表反映的是针对“未连接”这一事件的置信度。)*

---

### 2. 预测连接置信度（从高到低累计）

这部分数据与您提供的大表完全吻合，我为您抽离并计算了对应区间的详细表现：

| 预测连接置信度 | 累计样本数 | 累计平均预测置信度 | 累计真实连接发生率 | 差值 (预测-真实) | 累计 logloss |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **>= 95%** | 17 | 0.959687 | 0.882353 | 0.077334 | 0.401732 |
| **>= 90%** | 93 | 0.928303 | 0.731182 | 0.197121 | 0.752233 |
| **>= 85%** | 188 | 0.900336 | 0.670212 | 0.230123 | 0.815487 |
| **>= 80%** | 321 | 0.869408 | 0.607476 | 0.261931 | 0.866412 |
| **>= 75%** | 500 | 0.834590 | 0.514000 | 0.320590 | 0.933092 |
| **>= 70%** | 847 | 0.788579 | 0.407320 | 0.381259 | 0.976624 |