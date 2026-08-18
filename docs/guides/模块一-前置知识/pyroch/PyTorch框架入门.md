---
title: "🔥 PyTorch框架快速入门篇"
description: "PyTorch 是当前大模型训练和推理的事实标准框架"
pubDate: 2026-04-25
category: "prerequisites"
order: 400
chapter: 4
tags: ["PyTorch", "Tensor", "autograd", "计算图", "GPU"]
---

PyTorch 是当前大模型训练和推理的事实标准框架。本文聚焦于 AI Infra 工程师需要掌握的核心能力——从 Tensor 操作到完整训练循环，再到用 profiler 定位性能瓶颈，为后续的 CUDA 编程、分布式训练和推理优化打下基础。

<!-- more -->

## 📑 目录

- [1. Tensor：PyTorch 的基本数据单元](#1-tensorpytorch-的基本数据单元)
- [2. 自动微分（autograd）](#2-自动微分autograd)
- [3. nn.Module：模型的组织方式](#3-nnmodule模型的组织方式)
- [4. 完整训练循环](#4-完整训练循环)
- [5. GPU 训练基础](#5-gpu-训练基础)
- [6. 性能分析入门](#6-性能分析入门)
- [总结](#总结)
- [自我检验清单](#自我检验清单)
- [参考资料](#参考资料)

---

## 1. Tensor：PyTorch 的基本数据单元

如果把深度学习比作搭积木，Tensor 就是最基本的积木块。所有的模型参数、输入数据、中间计算结果，在 PyTorch 里都以 Tensor 的形式存在。

**正式定义**：Tensor（张量）是一个多维数组，可以看作 NumPy ndarray 的 GPU 加速版本，同时支持自动微分。

### 1.1 创建 Tensor 的常用方式

```python
import torch
import numpy as np

a = torch.tensor([1.0, 2.0, 3.0])          # 从 Python 列表创建
zeros = torch.zeros(3, 4)                    # 3x4 全零矩阵
ones = torch.ones(2, 3, 4)                   # 2x3x4 全一张量
rand_normal = torch.randn(3, 4)              # 标准正态分布
seq = torch.arange(0, 10, 2)                 # tensor([0, 2, 4, 6, 8])

x = torch.randn(3, 4)
y = torch.zeros_like(x)                      # 和 x 形状一样的全零 Tensor

t = torch.from_numpy(np.array([1.0, 2.0]))   # 从 NumPy 转换（共享内存）
```

### 1.2 形状操作：view, reshape, permute, squeeze, unsqueeze

把 Tensor 想象成一块可以任意捏形的橡皮泥——数据不变，只是换个排列方式。

```python
import torch

x = torch.arange(12)             # 一维，12 个元素
a = x.view(3, 4)                 # 变成 3x4（要求内存连续）
b = x.reshape(3, 4)              # 和 view 类似，但内存不连续时也能用
c = x.view(-1, 4)                # -1 自动推断为 3

# permute：交换维度，Transformer 中常用
t = torch.randn(2, 8, 12, 64)              # (batch, seq_len, heads, dim)
t_permuted = t.permute(0, 2, 1, 3)         # (batch, heads, seq_len, dim)

# squeeze/unsqueeze：去掉或插入大小为 1 的维度
e = torch.randn(1, 3, 1, 4)
print(e.squeeze().shape)                    # torch.Size([3, 4])
f = torch.randn(3, 4)
print(f.unsqueeze(0).shape)                 # torch.Size([1, 3, 4])
```

### 1.3 设备管理：CPU 与 GPU 数据搬运

```python
import torch

print(torch.cuda.is_available())            # GPU 是否可用

x = torch.randn(3, 4)
x_gpu = x.to('cuda')                        # CPU → GPU（推荐写法）
x_gpu = x.cuda()                             # 等价简写
x_cpu = x_gpu.cpu()                          # GPU → CPU

# 直接在 GPU 上创建（避免多余的搬运）
x_gpu = torch.randn(3, 4, device='cuda')
```

> **AI Infra 视角**：CPU-GPU 数据搬运走 PCIe 总线，带宽远低于 GPU 显存带宽。频繁的 `.cpu()` 和 `.cuda()` 调用往往是隐蔽的性能瓶颈，第 6 节会讲如何用 profiler 发现这类问题。

### 1.4 dtype 与内存：fp32, fp16, bf16

不同的数据精度直接决定显存占用和计算速度：

| dtype | 每元素字节 | 数值范围 | 使用场景 |
|-------|:-------:|---------|---------|
| float32 (fp32) | 4 | 大 | 默认精度，优化器状态 |
| float16 (fp16) | 2 | 小，易溢出 | 混合精度训练（需 Loss Scaling） |
| bfloat16 (bf16) | 2 | 与 fp32 相同 | 混合精度训练（推荐，Ampere+ GPU） |

> **白话理解**：fp32 像高清照片，又大又清晰；fp16 像压缩照片，体积减半但偶尔"失真"（溢出）；bf16 是聪明的压缩，体积和 fp16 一样小，但"失真"概率大大降低。

```python
import torch

x_fp32 = torch.randn(1000, 1000, dtype=torch.float32)   # 4 bytes/element
x_bf16 = torch.randn(1000, 1000, dtype=torch.bfloat16)  # 2 bytes/element
print(f"fp32: {x_fp32.nelement() * x_fp32.element_size() / 1024:.0f} KB")  # 3906 KB
print(f"bf16: {x_bf16.nelement() * x_bf16.element_size() / 1024:.0f} KB")  # 1953 KB

x = torch.randn(3, 4)                     # 默认 fp32
x_half = x.half()                          # 转 fp16
x_bf16 = x.to(torch.bfloat16)             # 转 bf16
```

---

## 2. 自动微分（autograd）

训练神经网络的核心是"根据损失调整参数"，而调整的依据就是梯度。PyTorch 的 autograd 引擎帮你自动完成这件事。

### 2.1 什么是计算图

> **比喻**：想象你在做一道菜。你把食材 A 和 B 混合得到 C，再把 C 加热得到 D（成品）。如果 D 味道不对（loss 太大），你需要反推——是 C 的问题？还是 A、B 的比例不对？计算图就是 PyTorch 帮你记下的"食谱"：它记录每一步操作，这样就能从结果反向推导出每种食材对结果的影响（梯度）。

**正式定义**：计算图（Computational Graph）是一个有向无环图（DAG），节点代表 Tensor，边代表运算操作。PyTorch 在前向传播时动态构建图，反向传播时沿图计算梯度。PyTorch 采用**动态计算图**（Define-by-Run），支持 if/else、for 循环等 Python 控制流。

### 2.2 requires_grad 和 backward()

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)  # 追踪所有操作
y = x * 3
z = y.sum()        # z = 3*2 + 3*3 = 15
z.backward()       # 反向传播
print(x.grad)      # tensor([3., 3.])，dz/dx = 3
```

要点：`backward()` 只能对标量调用；叶节点才保存梯度；计算图用完即释放。

### 2.3 梯度累积与清零

> **白话理解**：PyTorch 默认**累加**梯度，不自动清零。就像一个计数器，每次 `backward()` 都往上加。标准训练循环里，每个 step 开始前必须清零——否则上一轮梯度会混进来。

这个设计是有意为之的——梯度累积是显存不足时模拟大 batch 的常用技巧。

```python
import torch

x = torch.tensor([1.0], requires_grad=True)
(x * 2).sum().backward()
print(x.grad)         # tensor([2.])
(x * 3).sum().backward()
print(x.grad)         # tensor([5.]) ← 2+3，梯度被累加！
x.grad.zero_()        # 手动清零
(x * 4).sum().backward()
print(x.grad)         # tensor([4.]) ← 清零后正确
# 实际训练中用 optimizer.zero_grad() 一次性清零所有参数的梯度
```

### 代码示例：手动梯度下降学习 y = 2x + 1

```python
import torch

w = torch.tensor([0.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
x_train = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_train = torch.tensor([3.0, 5.0, 7.0, 9.0])

for epoch in range(100):
    loss = ((w * x_train + b - y_train) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= 0.01 * w.grad
        b -= 0.01 * b.grad
    w.grad.zero_()
    b.grad.zero_()

print(f"学到的模型: y = {w.item():.2f}x + {b.item():.2f}")
# 输出接近 y = 2.00x + 1.00
```

---

## 3. nn.Module：模型的组织方式

> **比喻**：如果 Tensor 是积木块，`nn.Module` 就是积木的"说明书"——它定义了积木怎么拼接（前向传播），并帮你清点所有零件（参数管理）。

### 3.1 定义模型：继承 nn.Module，实现 \_\_init\_\_ 和 forward

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

model = SimpleModel(784, 256, 10)
output = model(torch.randn(32, 784))    # 用 model(x)，不要直接调用 forward
print(output.shape)                       # torch.Size([32, 10])
```

### 3.2 参数管理：parameters(), named_parameters(), state_dict()

```python
import torch.nn as nn

model = nn.Linear(10, 5)
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")   # weight: [5, 10], bias: [5]

# 统计参数量（AI Infra 中非常常用）
total_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {total_params}")       # 55

# state_dict 用于保存和加载模型
print(model.state_dict().keys())       # odict_keys(['weight', 'bias'])
```

### 3.3 常用层

```python
import torch
import torch.nn as nn

# nn.Linear：大模型中用最多的层（Attention Q/K/V 投影、FFN）
linear = nn.Linear(512, 256)

# nn.Embedding：将 token id 映射为稠密向量（大模型第一层）
embedding = nn.Embedding(num_embeddings=50000, embedding_dim=768)
print(embedding(torch.tensor([0, 42])).shape)   # torch.Size([2, 768])

# nn.LayerNorm：Transformer 每个子层后都有，稳定训练
layer_norm = nn.LayerNorm(768)

# nn.Dropout：训练时随机丢弃神经元，model.eval() 后自动关闭
dropout = nn.Dropout(p=0.1)
```

### 代码示例：两层 MLP

```python
import torch
import torch.nn as nn

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

model = TwoLayerMLP(784, 256, 10)
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")  # 203,530
print(model(torch.randn(64, 784)).shape)  # torch.Size([64, 10])
```

---

## 4. 完整训练循环

训练循环是 AI Infra 工程师每天都要接触的代码——即使在分布式场景下，核心循环依然是同一个模式。

### 4.1 数据加载：Dataset 和 DataLoader

- **Dataset**：定义"数据集里有什么"和"怎么取一条数据"
- **DataLoader**：定义"怎么分 batch 喂给模型"（batching、shuffling、多进程加载）

```python
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data, self.labels = data, labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = SimpleDataset(torch.randn(1000, 784), torch.randint(0, 10, (1000,)))
loader = DataLoader(dataset, batch_size=64, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)
```

> **AI Infra 视角**：`num_workers` 过小会导致 GPU 等数据（data loading bottleneck）；`pin_memory=True` 使传输走异步 DMA，避免 CPU 拷贝开销。

### 4.2 标准训练流程

每个 step 的核心五步：forward → loss → backward → optimizer.step → zero_grad。

### 4.3 学习率调度

```python
import torch

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# 常用调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# 每个 epoch 结束后调用 scheduler.step()
```

### 4.4 Checkpoint 保存与恢复

```python
import torch

# 保存（模型 + 优化器 + 训练进度，断点恢复需要全部保存）
torch.save({
    'epoch': epoch, 'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss,
}, 'checkpoint.pt')

# 加载
ckpt = torch.load('checkpoint.pt', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
```

### 代码示例：完整 MNIST 训练脚本

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size, lr, num_epochs = 128, 1e-3, 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=2, pin_memory=True)

# 模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256),
                                 nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 10))
    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# 训练
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    scheduler.step()

    # 验证
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f} "
          f"| Acc: {correct/len(test_set)*100:.2f}%")

    # 保存 checkpoint
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, f'ckpt_ep{epoch+1}.pt')
```

---

## 5. GPU 训练基础

### 5.1 将模型和数据搬到 GPU

```python
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Linear(100, 10).to(device)
x = torch.randn(32, 100).to(device)     # 模型和数据必须在同一设备上
y = model(x)
```

### 5.2 混合精度训练入门

核心思路：前向和反向用低精度（fp16/bf16）加速计算，参数更新用高精度（fp32）保证精度。

> **白话理解**：算草稿用铅笔（低精度，快），写定稿用钢笔（高精度，准）。关键的参数更新步骤用高精度保证不丢信息，其他大量计算用低精度跑得更快。

```python
import torch
import torch.nn as nn

device = torch.device('cuda')
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# BF16 混合精度（推荐，Ampere+ GPU，不需要 GradScaler）
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# FP16 混合精度（需要 GradScaler 防止梯度下溢）
scaler = torch.cuda.amp.GradScaler()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        loss = nn.functional.cross_entropy(model(images), labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 5.3 显存查看

```python
import torch

print(f"已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"峰值:   {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print(torch.cuda.memory_summary(abbreviated=True))

# 分析某段代码的显存
torch.cuda.reset_peak_memory_stats()
# ... 运行目标代码 ...
print(f"峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

### 代码示例：混合精度 vs fp32 显存对比

```python
import torch
import torch.nn as nn

assert torch.cuda.is_available(), "需要 GPU"
device = torch.device('cuda')
criterion = nn.CrossEntropyLoss()

# batch 要足够大，让激活值成为显存主项。混合精度不会缩小 FP32 权重和优化器状态，
# batch 太小时这部分固定开销会完全掩盖激活值的收益，甚至出现"负节省"
BATCH = 65536
data = torch.randn(BATCH, 1024, device=device)
labels = torch.randint(0, 10, (BATCH,), device=device)

def make_model():
    return nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(),
                         nn.Linear(2048, 1024), nn.ReLU(),
                         nn.Linear(1024, 10)).to(device)

def step_peak(amp_dtype=None):
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if amp_dtype is None:
        loss = criterion(model(data), labels)
    else:
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            loss = criterion(model(data), labels)
    loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    # 关键：必须真正释放本轮的模型和优化器状态，否则会叠加到下一轮的峰值里
    del model, opt, loss
    torch.cuda.empty_cache()
    return peak

fp32_peak = step_peak(None)
bf16_peak = step_peak(torch.bfloat16)

print(f"FP32: {fp32_peak:.1f} MB | BF16: {bf16_peak:.1f} MB | 节省: {(1-bf16_peak/fp32_peak)*100:.1f}%")
# 参考输出：FP32: 1881.6 MB | BF16: 1246.6 MB | 节省: 33.7%
```

⚠️ **两个容易踩的坑**：

1. `torch.cuda.reset_peak_memory_stats()` 只重置峰值计数器，**不释放**已占用的显存。如果测 BF16 时上一轮的 FP32 模型和优化器状态还活着，它们会被算进 BF16 的峰值，结果会出现节省为负。必须 `del` 掉对象再 `empty_cache()`。
2. 混合精度省的是**激活值**（以及部分中间结果），权重和 AdamW 的一阶/二阶动量仍是 FP32。模型小、batch 小的时候这部分固定开销占绝对主导，测出来会是"节省 0%"——需要把 batch/序列长度放大到激活值占主导，才能看到预期的约 30%~50% 节省。

---

## 6. 性能分析入门

写出能跑的训练代码只是第一步，跑得快才是 AI Infra 的核心追求。`torch.profiler` 能告诉你每个操作花了多少时间、GPU 利用率如何、哪里在空等数据。

### 6.1 torch.profiler 基本用法

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = torch.nn.Linear(1024, 1024).cuda()
x = torch.randn(64, 1024).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True) as prof:
    with record_function("forward_pass"):
        output = model(x)
    with record_function("backward_pass"):
        output.sum().backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 6.2 如何读懂 profiler 输出

| 列名 | 含义 | 关注点 |
|------|------|--------|
| Name | 操作名称 | 找到耗时最多的操作 |
| CPU total | CPU 端总耗时 | 远大于 CUDA total 说明存在 CPU 瓶颈 |
| CUDA total | GPU 端总耗时 | 核心指标 |
| # of Calls | 调用次数 | 高频小操作可能累积成瓶颈 |

常见性能问题模式：

1. **CPU 时间远大于 CUDA 时间**：GPU 在等 CPU（数据预处理、Python 开销）
2. **大量小 CUDA kernel**：launch overhead 累积，考虑 `torch.compile` 或算子融合
3. **频繁 CPU-GPU 同步**：`.item()`、`print(tensor)` 会触发同步，阻塞 GPU 流水线

### 6.3 常见性能问题

**数据加载瓶颈**：增加 `num_workers`，使用 `prefetch_factor` 预取数据。

**CPU-GPU 数据搬运**：避免循环中反复 `.cuda()`，一次性搬运到 GPU。

**隐式同步**：`loss.item()` 触发 CPU-GPU 同步，应每 N 步才记录一次。

```python
# 不好：每步都同步
for batch in dataloader:
    loss = train_step(batch)
    print(f"loss: {loss.item()}")       # 每步同步！

# 好：每 100 步记录一次
for i, batch in enumerate(dataloader):
    loss = train_step(batch)
    if i % 100 == 0:
        print(f"step {i}, loss: {loss.item()}")
```

### 代码示例：用 profiler 分析一个训练 step

```python
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(),
                      nn.Linear(4096, 4096), nn.ReLU(),
                      nn.Linear(4096, 10)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
data = torch.randn(128, 1024, device=device)
labels = torch.randint(0, 10, (128,), device=device)

# warmup（避免首次调用的初始化开销）
for _ in range(3):
    criterion(model(data), labels).backward()
    optimizer.step(); optimizer.zero_grad()

# profiler 分析
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True) as prof:
    with record_function("data_to_gpu"):
        d = torch.randn(128, 1024).to(device)
        l = torch.randint(0, 10, (128,)).to(device)
    with record_function("forward"):
        loss = criterion(model(d), l)
    with record_function("backward"):
        loss.backward()
    with record_function("optimizer_step"):
        optimizer.step(); optimizer.zero_grad()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
prof.export_chrome_trace("trace.json")   # 可在 chrome://tracing 中可视化
```

从输出中可以快速判断：backward 是否比 forward 慢（正常约 2 倍），数据搬运是否占过多时间，优化器步骤是否有异常耗时。

---

## 📝 总结

| 层次 | 内容 | 核心能力 |
|------|------|---------|
| Tensor | 数据表示与操作 | 形状操作、设备管理、dtype 选择 |
| autograd | 自动微分 | 计算图、梯度计算、梯度清零 |
| nn.Module | 模型组织 | 定义模型、参数管理、常用层 |
| 训练循环 | 端到端训练 | DataLoader、训练流程、checkpoint |
| GPU 训练 | 硬件加速 | 混合精度、显存管理 |
| 性能分析 | 瓶颈定位 | profiler 使用、常见瓶颈识别 |

这些构成了理解后续分布式训练、CUDA 编程和推理优化的必要基础。掌握这些内容后，你在阅读 DeepSpeed、Megatron-LM 等框架的代码时，不会因为 PyTorch 基础不牢而卡住。

---

## 🎯 自我检验清单

完成本文学习后，你应该能够：

- 能用 `torch.randn`、`torch.zeros` 等方法创建任意形状的 Tensor，并熟练使用 `view`、`permute`、`squeeze` 等操作变换形状
- 能解释 `requires_grad=True` 的作用，手动构建计算图并调用 `backward()` 获取梯度
- 能解释为什么每个训练 step 都需要调用 `optimizer.zero_grad()`，以及梯度累积的原理
- 能继承 `nn.Module` 实现自定义模型，并使用 `parameters()` 和 `state_dict()` 管理参数
- 能写出完整的训练循环：DataLoader → forward → loss → backward → optimizer.step → checkpoint
- 能使用 `torch.autocast` 实现 bf16/fp16 混合精度训练，并解释 `GradScaler` 的作用
- 能使用 `torch.profiler` 分析训练 step，读懂输出并识别数据加载和 CPU-GPU 传输瓶颈
- 能解释 fp32、fp16、bf16 三种精度的区别及选择依据

## 📚 参考资料

- [PyTorch 官方文档](https://pytorch.org/docs/stable/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch autograd 机制详解](https://pytorch.org/docs/stable/notes/autograd.html)
- [torch.profiler 文档](https://pytorch.org/docs/stable/profiler.html)
- [Automatic Mixed Precision (AMP) 文档](https://pytorch.org/docs/stable/amp.html)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
