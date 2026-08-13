---
title: "第9章：性能分析与 Benchmark"
description: "从指标定义、负载生成、时间线分析到公开基准与 CI 门禁，建立可解释、可复现的推理性能工程方法"
pubDate: 2026-08-12
category: "inference-optimization"
order: 38
tags: ["性能分析", "Benchmark", "vLLM Bench", "GenAI-Perf", "性能回归"]
---

前面的章节回答了“推理系统能做哪些优化”，这一章回答一个更基础的问题：**你怎么证明优化真的有效？**

“吞吐提升了”并不是完整结论。

它可能以更差的首 Token 延迟为代价，也可能来自更短的输出长度，甚至只是因为压测客户端没有把服务压满。

本章会把一次 Benchmark 拆成五个可以审查的环节：定义指标、生成负载、分析时间线、比较公开结果、设置回归门禁。

<!-- more -->

## 📑 本章路线

- [9.1 推理指标体系](/AIInfraGuide/inference/模块四-推理优化/第9章-性能分析与benchmark/91-推理指标体系)：先把 TTFT、TPOT、ITL、E2E、throughput、goodput 与 percentile 讲清楚
- [9.2 压测工具](/AIInfraGuide/inference/模块四-推理优化/第9章-性能分析与benchmark/92-压测工具)：用 `vllm bench`、`benchmark_serving.py` 与 GenAI-Perf/AIPerf 产生可控负载
- [9.3 性能分析工具](/AIInfraGuide/inference/模块四-推理优化/第9章-性能分析与benchmark/93-性能分析工具)：从 PyTorch Profiler 下钻到 Nsight Systems
- [9.4 权威基准](/AIInfraGuide/inference/模块四-推理优化/第9章-性能分析与benchmark/94-权威基准)：学会公平阅读 MLPerf 等公开结果
- [9.5 性能回归门禁](/AIInfraGuide/inference/模块四-推理优化/第9章-性能分析与benchmark/95-性能回归门禁)：把可复现测量接入 CI，而不是靠人盯表格

---

## 1. Benchmark 不是“跑一个命令”

一个可信的性能结论至少包含四件事：

1. **工作负载**：输入多长、输出多长、如何到达、是否流式。
2. **被测系统**：模型、引擎、精度、硬件、并行方式与配置。
3. **统计口径**：从哪里计时、如何聚合、失败请求是否计入。
4. **约束条件**：质量阈值、延迟 SLO、功耗或成本上限。

缺少其中任何一项，数字都很难复现。

例如“两张卡达到 1,000 tok/s”仍然有大量空白：

- 是输入 Token、输出 Token，还是二者之和？
- 模型与量化精度是什么？
- 请求是同时灌入，还是按到达率发送？
- 平均输出长度是多少？
- 是否有请求失败或提前结束？
- TTFT P99 是否已经不可接受？

📌 **关键点**：Benchmark 的产物不是一个数字，而是一份“条件—结果—解释”三者绑定的实验记录。

---

## 2. 本章贯穿的心智模型

把在线推理想成一家餐厅：

- 请求到达相当于顾客进店。
- 排队等待相当于调度队列。
- Prefill 相当于第一次备菜。
- 第一个 Token 相当于第一道菜上桌。
- Decode 相当于后续菜品逐道上桌。
- 最后一个 Token 到达才算整单完成。

于是：

- **TTFT** 关心“多久吃到第一口”。
- **ITL** 关心“相邻两口之间等多久”。
- **TPOT** 关心“后续每口平均等多久”。
- **E2E** 关心“整顿饭多久结束”。
- **throughput** 关心“全店单位时间做了多少”。
- **goodput** 只数“满足承诺的有效订单”。

这个类比也暴露了 trade-off：

为了让全店总出餐量更大，可以把更多订单攒成批。

但等待成批会增加个别顾客的延迟。

所以吞吐与延迟必须放在同一张报告中。

---

## 3. 一次完整实验的六步法

### 第一步：写假设

不要先改参数再找故事。

先写一句可以被证伪的话：

> 在固定模型、输入/输出长度分布和请求到达率下，增大调度 Token 预算会提升输出吞吐，同时不使 TTFT P95 超过既定 SLO。

这句话同时给出了：

- 自变量：调度 Token 预算。
- 控制变量：模型、长度分布、到达率。
- 目标指标：输出吞吐。
- 护栏指标：TTFT P95。

### 第二步：冻结环境

记录不可省略的实验元数据：

```yaml
experiment_id: chapter9-example
timestamp_utc: "<填写实际时间>"
git_commit: "<填写实际 commit>"
model_id: "<仓库或本地模型标识>"
model_revision: "<固定 revision>"
tokenizer_revision: "<固定 revision>"
engine: "vllm"
engine_version: "<填写实际版本>"
container_image_digest: "<填写 sha256>"
gpu_model: "<填写实际型号>"
gpu_count: "<填写实际数量>"
driver_version: "<填写实际版本>"
cuda_version: "<填写实际版本>"
precision: "<bf16/fp16/fp8/int8/...>"
tensor_parallel_size: "<填写>"
server_args: ["<完整参数>"]
dataset_sha256: "<填写>"
random_seed: 2026
```

不要把 `<填写实际版本>` 替换成猜测值。

### 第三步：定义流量

在线服务常见两种负载：

- **闭环并发**：保持 N 个在途请求，一个结束再补一个。
- **开环到达率**：请求按指定速率到达，不因服务变慢而自动降速。

闭环适合寻找饱和吞吐。

开环更接近用户独立到达的生产流量，也更容易暴露排队雪崩。

两者回答的问题不同，不能把结果直接横比。

### 第四步：预热

首次请求可能包含：

- 权重懒加载。
- CUDA Context 初始化。
- `torch.compile` 编译。
- CUDA Graph 捕获。
- 内存池扩展。
- Prefix Cache 冷启动。

预热不是“删掉不好看的数据”，而是区分冷启动与稳态。

如果业务关心扩容后的首批请求，冷启动本身就应单独测量。

### 第五步：重复与保存原始数据

至少保存：

- 每个请求的开始、首 Token、结束时间。
- 输入/输出 Token 数。
- 状态码与错误原因。
- 聚合前的原始样本。
- 服务端日志和监控快照。
- 压测命令及配置文件。

只保存最终均值，之后无法重算 P95，也无法排查异常请求。

### 第六步：解释而非宣判

正确表达：

> 在本实验配置下，候选版本的输出吞吐更高；TTFT P95 仍在预先声明的阈值内。该结论尚未覆盖长上下文与突发流量。

危险表达：

> 候选版本性能更好。

后者把局部样本错误外推成普遍规律。

---

## 4. 贯穿本章的实验矩阵

入门者最容易一次改太多变量。

建议从一个小矩阵开始：

| 维度 | 起步选择 | 后续扩展 |
|---|---|---|
| 输入长度 | 固定短、固定长 | 生产分布 |
| 输出长度 | 固定长度 | 多峰分布 |
| 到达模式 | 低负载开环 | 梯度升压、突发 |
| 并发 | 1、低、中、高 | 找拐点 |
| 缓存 | 明确关闭或冷缓存 | 热缓存命中率分层 |
| 精度 | 生产默认精度 | 量化候选 |

每次只改变一个主变量。

如果必须同时改变多个参数，就把它当成“整体方案 A/B”，不要声称某个单独参数导致了结果。

---

## 5. 结果报告的最小模板

```markdown
# 实验结论

## 假设
<可证伪的一句话>

## 被测版本
- baseline: <commit/image/config>
- candidate: <commit/image/config>

## 固定条件
- hardware:
- model and revision:
- tokenizer and revision:
- dataset and hash:
- traffic model:
- input/output length distribution:
- warmup and repetitions:

## 原始指标
- success/error/cancel count:
- TTFT P50/P95/P99:
- TPOT P50/P95/P99:
- ITL P50/P95/P99:
- E2E P50/P95/P99:
- request/output-token throughput:
- goodput under declared SLO:

## 解释
- 支持或否定假设的证据：
- trade-off：
- 未覆盖范围：
- 下一步：
```

这里故意没有填任何示例性能数值。

模板中的数字必须来自你的实际运行产物。

---

## 6. 三类最常见的误判

### 误判一：均值稳定，所以尾延迟稳定

平均值会隐藏少量极慢请求。

交互式应用中，P95/P99 往往比均值更接近用户投诉。

但高分位数也需要足够样本；样本太少时，P99 可能只是最大值的别名。

### 误判二：GPU 利用率高，所以服务健康

高利用率只说明 GPU 忙。

它可能在做有效计算，也可能在重算、等待通信、执行低效 Kernel。

必须结合吞吐、延迟、显存、时间线与错误率判断。

### 误判三：Token throughput 高，所以用户体验好

更长输出天然贡献更多输出 Token。

只比较 tok/s 而不固定长度分布，会奖励“生成得更长”的系统。

即使长度固定，高吞吐也可能来自激进组批，并伴随 TTFT 尾延迟恶化。

---

## 7. 工具在证据链中的位置

`vllm bench serve` 和 GenAI-Perf 回答：

> 外部客户端观察到什么？

PyTorch Profiler 回答：

> 框架与算子时间花在哪里？

Nsight Systems 回答：

> CPU、CUDA、通信和 Kernel 在时间线上如何交错？

公开 Benchmark 回答：

> 在一套共同规则下，不同系统的结果如何比较？

CI 回归门禁回答：

> 如何防止已经获得的性能被后续改动悄悄破坏？

这些工具不是替代关系，而是从现象到根因的证据链。

---

## 8. 学习顺序建议

第一次阅读时按 9.1 → 9.2 → 9.3 顺序。

先理解计时口径，再运行工具，否则很容易“会敲命令、不会读结果”。

当你能稳定产出一份本地实验报告后，再读 9.4。

最后把稳定场景固化为 9.5 的回归门禁。

不要一开始就在共享 CI GPU 上追求极窄阈值。

先测量环境噪声，再决定门禁精度。

---

## 📝 总结

- Benchmark 是“工作负载、系统、统计口径、约束条件”共同定义的实验。
- 延迟与吞吐存在 trade-off，任何单指标冠军都可能掩盖代价。
- 开环与闭环负载回答不同问题，不能混为一谈。
- 原始样本、完整元数据和未覆盖范围与最终数字同样重要。
- 压测、Profiler、公开基准和 CI 门禁组成从测量到治理的完整闭环。

## 🎯 自我检验清单

- 能否解释为什么“tok/s 更高”不等于“用户体验更好”？
- 能否写出一个包含自变量、控制变量、目标指标和护栏指标的假设？
- 能否区分开环到达率和闭环并发？
- 能否说明预热数据为什么不能与稳态数据混算？
- 能否列出复现实验所需的模型、软件、硬件与数据元信息？
- 能否在结论中明确适用范围，而不是把一次测试外推到所有场景？

## 📚 官方参考

- [vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- [NVIDIA GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html)
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [MLPerf Inference Submission Guide](https://docs.mlcommons.org/inference/submission/)
