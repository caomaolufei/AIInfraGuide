---
title: "第8章：性能分析与 Benchmark"
description: "建立完整的推理性能指标体系，掌握 vllm bench、GenAI-Perf 等压测工具和性能分析工具，制定性能回归门禁"
pubDate: 2026-04-16
category: "inference-optimization"
order: 38
tags: ["性能分析", "Benchmark", "vLLM Bench", "GenAI-Perf", "性能回归"]
---

## 本章简介

优化效果需要量化验证，本章建立推理场景的完整性能评估体系，压测工具以 vLLM 自带的基准工具为主。

**推理指标体系**定义完整指标集：QPS、TTFT(P50/P95)、TPOT(P50/P95)、Throughput、显存峰值、GPU 利用率，分析指标间的关联与 trade-off，强调单一指标无法反映全貌。

**压测工具**覆盖 vLLM 自带的 `vllm bench serve` / `vllm bench latency` / `vllm bench throughput`、GenAI-Perf（LLM 指标一站式输出）、Triton Perf Analyzer，以及自定义压测脚本的设计要点和可复现的 Benchmark 配置管理。

**性能分析工具**在推理场景下的最佳实践：torch.profiler 算子级分析、Nsight Systems 全链路分析、Nsight Compute Kernel 级下钻。

**权威基准**介绍 MLPerf Inference（Datacenter）和 LLM Perf 评测趋势。

**性能回归门禁**制定规则（如 TPOT P95 退化 > 5% 则 Block Merge）、CI 自动化集成和退化定位方法（git bisect + Nsight 对比）。

**动手实验**：用 `vllm bench serve` 输出完整指标报告，模拟性能退化并用 Nsight Systems 定位。

## 本章小节

- **9.1 推理指标体系**：QPS、TTFT、TPOT、Throughput、显存与利用率的关联
- **9.2 压测工具**：vllm bench、GenAI-Perf、可复现的压测配置
- **9.3 性能分析工具**：torch.profiler、Nsight Systems/Compute
- **9.4 权威基准**：MLPerf Inference 与评测趋势
- **9.5 性能回归门禁**：门禁规则、CI 集成、退化定位
