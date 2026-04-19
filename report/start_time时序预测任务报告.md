## start_time时序预测任务

start_time 时序预测旨在根据用户历史使用行为推断其下一次启动时间，但该任务面临行为不确定性挑战。用户在短时间尺度内可能呈现相对稳定的高频使用模式，而在跨天或长时间间隔场景下，其行为往往具有突发性和弱规律性特征。由此导致模型在长间隔预测上的误差显著放大，成为影响整体预测准确率的核心瓶颈。

### 1. 数据集分析

（1）总共有131名用户共85,538条数据，每名用户具体的样本数量不一，样本数量与对应的用户数量统计信息如下图1所示。

<img src="./imgs/报告/image-20260125000101427.png" alt="image-20260125000101427" style="zoom:40%;" />

​																							图1

**问题1：**131名用户中样本数在0-100的有43名用户，样本数在100以上的有88名用户。小样本数量用户（样本数0-100）占32.8%，样本数大于500的用户只有45名占到34.4%。

（2）以用户103811283287015454的数据（共649条样本数）为例分析，其前200条数据对应的start_time的间隔如下图2所示。可以看到其间隔波动较大其规律性较差，间隔最小以秒为单位，间隔最大达到56499分钟。

<img src="./imgs/报告/image-20260125005128835.png" alt="image-20260125005128835" style="zoom:40%;" />

​                                                                                               图2

再以用户102529731903094907的数据进行分析，其在2025年8月1日到8月22日期间几乎每天使用，且每一天使用频率也相对较高。但8月22日的最后一下记录的下一条记录是在9月2日，跨度突然且很大。用户在整个9月只有八天使用，十月份只有3天使用，而且间隔很不规律。这是预测任务中的一个难点。

**问题2：**一些用户虽然有相对多的数据量，但数据波动性较大且规律性较差。

### 2.现有时序预测方法对比

尝试使用主流的基于深度学习的PatchTST[1]和Timer[2]以及基于机器学习的XgBoost[3]和CatBoost[4]对样本数量大于100的用户（88名）上进行下一个start_time的任务预测。4个baselines以及我们的方法在88名用户上的测试集上的平均准确率如下表所示（正确标准是预测值与真实值的差值在20min内）。

| 模型       | PatchTST | Timer | XgBoost | CatBoost | ours   |
| ---------- | -------- | ----- | ------- | -------- | ------ |
| 平均准确率 | 57.84%   | 9.69% | 58.23%  | 58.02%   | 70.72% |

（1）可以看到Timer表现不佳，PatchTST/XgBoost/CatBoost表现接近。由于数据序列长度不足，且用户数据差异较大，Timer模型无法泛化出有效的用户行为模式，容易陷入过拟合。 从实验结果来看，当单用户数据量在1k以上时Timer的预测准确率最高达到了65.22%，然而对大量低数据量用户的测试准确率却趋近于0。

（2）进一步分析XgBoost的预测表现如下图3所示，XgBoost对各个用户预测的准确率集中在50-70%这个区间（46.6%），准确率低于50%的占26.1%。分析**这部分用户的数据对应上一部分所说的问题1和问题2，即样本数量小或样本数数量相对多但用户行为规律性较差。**

<img src="./imgs/报告/image-20260125185455754.png" alt="image-20260125185455754" style="zoom:45%;" />

​                                                                                       图3

（3）进一步分析XgBoost预测正确和预测错误的部分发现，XgBoost对于时间间隔本身较小的数据有较好的预测表现，但难以在下一个start_time相较于当前start_time是一个长间隔时表现不佳。这不仅仅是XgBoost面临的问题，当前所有尝试的baselines都面临了相似的问题。

**总结：**（1）实验结果表明，下一个 start_time 的预测准确率整体受限，主要源于用户使用行为在时间维度上的不稳定性。用户在短时间内可能呈现高频、近似规律的使用模式，但在跨天或长时间间隔场景下，其行为往往表现为突发性和不规则性，难以由历史时间间隔或局部上下文有效刻画。由于长间隔对预测误差和准确率评估影响显著，该类行为的不确定性成为限制模型性能提升的主要因素。

（2）进一步而言，若将预测目标从“下一次 start_time”扩展为“下一天所有的 start_time”，该任务仍然面临困难。一旦在（1）所述的关键的长时间间隔节点上发生预测偏差，误差将沿时间轴逐步累积，并放大后续时间点的预测不确定性，从而使多步预测难以获得稳定且可靠的结果。

<div style="font-size: 12px;">[1] Nie, Y., Nguyen, N.H., Sinthong, P., & Kalagnanam, J. (2022). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ArXiv, abs/2211.14730.

<div style="font-size: 12px;">[2] Yong Liu, Haoran Zhang, Chenyu Li, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. 2024. Timer: generative pre-trained transformers are large time series models. In Proceedings of the 41st International Conference on Machine Learning (ICML'24), Vol. 235. JMLR.org, Article 1313, 32369–32399.

<div style="font-size: 12px;">[3] Chen Shen. 2025. Stock forecasts based on the XGBoost model. In Proceedings of the 2024 5th International Conference on Computer Science and Management Technology (ICCSMT '24). Association for Computing Machinery, New York, NY, USA, 1198–1202. 

<div style="font-size: 12px;">[4] Lijie Zhang and Dominik Jánošík. Enhanced Short-Term Load Forecasting with Hybrid Machine Learning Models: CatBoost and XGBoost Approaches. Expert Systems with Applications 241 (2024), 122686.