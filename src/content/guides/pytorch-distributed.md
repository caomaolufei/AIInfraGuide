---
title: "PyTorch 分布式训练入门"
description: "使用 PyTorch DDP 和 FSDP 进行分布式训练的实践指南"
pubDate: 2024-03-01
category: "training-frameworks"
order: 1
tags: ["PyTorch", "分布式训练", "DDP"]
---

## 为什么需要分布式训练？

随着模型规模的增长，单 GPU 已无法满足训练需求。分布式训练通过多 GPU / 多节点协同工作来加速训练过程。

### 数据并行 vs 模型并行

- **数据并行**（Data Parallelism）：每个 GPU 持有完整模型副本，处理不同数据批次
- **模型并行**（Model Parallelism）：将模型拆分到不同 GPU
- **流水线并行**（Pipeline Parallelism）：将模型按层拆分，形成流水线

## PyTorch DDP

DistributedDataParallel（DDP）是 PyTorch 中最常用的数据并行方案：

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train(rank, world_size):
    setup(rank, world_size)

    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    dist.destroy_process_group()
```

### 启动方式

```bash
# 使用 torchrun 启动 4 GPU 训练
torchrun --nproc_per_node=4 train.py
```

## PyTorch FSDP

FullyShardedDataParallel（FSDP）将模型参数、梯度和优化器状态分片到多个 GPU：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    MyModel(),
    auto_wrap_policy=size_based_auto_wrap_policy,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    ),
)
```

### DDP vs FSDP 对比

| 特性 | DDP | FSDP |
|------|-----|------|
| 显存效率 | 低（完整模型副本） | 高（参数分片） |
| 通信开销 | AllReduce 梯度 | AllGather + ReduceScatter |
| 适用规模 | 中小模型 | 大模型 |
| 实现复杂度 | 低 | 中 |

## 总结

DDP 适合大多数常规训练场景，FSDP 则在训练大模型时能显著降低显存需求。根据模型规模和硬件条件选择合适的策略。
