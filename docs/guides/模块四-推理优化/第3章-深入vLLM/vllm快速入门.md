---
title: "3.1 vLLM 快速入门：从可复现实验到在线服务"
description: "用固定版本跑通离线与在线推理，理解请求、KV Cache 与批处理，并建立从安装到性能排障的完整实验闭环"
pubDate: 2026-04-15
category: "inference-optimization"
order: 301
chapter: 3
tags: ["vLLM", "LLM推理", "PagedAttention", "模型部署", "推理优化"]
---

## 目录

- [1. 真实问题：能生成不等于能服务](#1-真实问题能生成不等于能服务)
- [2. 先用白话建立直觉](#2-先用白话建立直觉)
- [3. 固定实验边界](#3-固定实验边界)
- [4. 安装与启动前体检](#4-安装与启动前体检)
- [5. 实验一：离线批量推理](#5-实验一离线批量推理)
- [6. 实验二：OpenAI 兼容在线服务](#6-实验二openai-兼容在线服务)
- [7. 从源码看最短调用链](#7-从源码看最短调用链)
- [8. 手算一次显存与张量形状](#8-手算一次显存与张量形状)
- [9. 实验三：观察连续批处理](#9-实验三观察连续批处理)
- [10. 实验四：验证 Prefix Cache](#10-实验四验证-prefix-cache)
- [11. 指标应该怎样解释](#11-指标应该怎样解释)
- [12. 失败模式与分层排障](#12-失败模式与分层排障)
- [13. 第一轮参数取舍](#13-第一轮参数取舍)
- [14. 自检](#14-自检)
- [参考资料](#参考资料)

---

## 1. 真实问题：能生成不等于能服务

在笔记本里，`transformers.generate()` 已经可以回答问题。为什么还需要 vLLM？

假设一张 GPU 上同时到达三类请求：

- A：32-token Prompt，生成 1,000 token；
- B：8,000-token Prompt，只生成 16 token；
- C：在 A 生成到第 100 个 token 时到达。

普通静态 batch 面临一个两难：

1. 等一批所有序列完成，短请求被长请求拖住；
2. 每个请求各跑一个模型副本，权重与 KV Cache 很快耗尽显存。

线上服务还要回答更多问题：

- C 能否在 A 结束前插入执行？
- B 的长 Prefill 会不会让 A 的流式输出卡住？
- 客户端断开后，已经占用的 KV block 何时释放？
- 两个请求共享 2,000-token system prompt 时，能否避免重复计算？
- 吞吐提高时，P99 TTFT 是否已经超过 SLO？

vLLM 首先是一个**有状态的推理调度系统**，其次才是一个调用高性能 Kernel 的模型封装。

## 2. 先用白话建立直觉

把 GPU 想成厨房，模型权重是固定灶台，KV Cache 是每桌正在使用的餐具。

`transformers.generate()` 更像“这一桌吃完，再接下一桌”。vLLM 则每一小轮重新排单：

1. 给正在生成的请求各算少量 token；
2. 用剩余预算处理新请求的一段 Prompt；
3. 请求结束立即回收它的 KV block；
4. 下一轮重新组成 batch。

这叫 continuous batching。它不是把“batch size”固定为一个数字，而是让 batch 成员随 step 改变。

PagedAttention 解决的是餐具摆放问题。KV Cache 不要求每个请求占用一段按最大长度预留的连续显存，而是按固定 token block 分配。请求持有的是逻辑位置到物理 block 的映射。

两个概念不要混淆：

- continuous batching 决定**本轮算谁**；
- 分页 KV Cache 决定**历史状态放哪**。

## 3. 固定实验边界

本文所有版本敏感事实与源码路径固定到：

```text
vLLM tag: v0.27.1
Git tag: https://github.com/vllm-project/vllm/tree/v0.27.1
```

先记录三类版本：

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import vllm; print(vllm.__version__)"
nvidia-smi
```

不要把 `latest` 文档、旧博客和本机其他版本的字段混在一起。本文实验使用：

```bash
export MODEL=Qwen/Qwen2.5-0.5B-Instruct
export PORT=8000
```

0.5B 模型便于跑通流程。它的绝对性能不能外推到 7B、70B 或 MoE 模型。

### 3.1 可复现记录

每次实验至少保存：

```text
vLLM / PyTorch / CUDA / driver 版本
GPU 型号、数量、互联与功耗限制
模型仓库与 revision
完整启动命令
输入、输出 token 长度分布
并发、request rate、warmup 次数
成功率、TTFT、TPOT、ITL、吞吐
```

“H100 上 10k tokens/s”不是可复现结论，因为缺少模型、长度和并发。

## 4. 安装与启动前体检

### 4.1 平台边界

NVIDIA GPU 的常规部署目标是 Linux。macOS 适合编辑客户端代码，但不能把 Apple GPU 当作 CUDA 环境照抄下面的服务端实验。

先确认驱动可见：

```bash
nvidia-smi
```

若容器内运行，再确认设备透传：

```bash
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi
```

镜像 tag 只是示例；生产环境应锁定团队验证过的 CUDA 基础镜像。

### 4.2 建立隔离环境

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install "vllm==0.27.1" --torch-backend=auto
```

验证导入与命令入口：

```bash
python -c "import vllm; assert vllm.__version__ == '0.27.1'"
vllm serve --help > /tmp/vllm-serve-help.txt
vllm bench serve --help > /tmp/vllm-bench-help.txt
```

若 `vllm` 与 `python` 指向不同虚拟环境：

```bash
which python
which vllm
python -m pip show vllm
```

### 4.3 下载与权限

对需要授权的 Hugging Face 模型：

```bash
huggingface-cli login
```

固定模型 revision 的 Python 入口：

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    revision="main",  # 生产环境替换为实际 commit SHA
)
```

`main` 仍会漂移。正式基线应记录 commit SHA，而不是只记录模型名。

## 5. 实验一：离线批量推理

新建临时脚本 `offline_demo.py`：

```python
from time import perf_counter

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

messages = [
    [{"role": "user", "content": "用两句话解释 KV Cache。"}],
    [{"role": "user", "content": "Prefill 和 Decode 有什么区别？"}],
    [{"role": "user", "content": "为什么在线推理需要调度器？"}],
]

params = SamplingParams(
    temperature=0,
    max_tokens=64,
    seed=7,
)

t0 = perf_counter()
llm = LLM(
    model=MODEL,
    generation_config="vllm",
    max_model_len=2048,
)
loaded = perf_counter()
outputs = llm.chat(messages, params)
finished = perf_counter()

print(f"load_seconds={loaded - t0:.3f}")
print(f"generate_seconds={finished - loaded:.3f}")
for i, output in enumerate(outputs):
    candidate = output.outputs[0]
    print(
        i,
        f"prompt_tokens={len(output.prompt_token_ids)}",
        f"output_tokens={len(candidate.token_ids)}",
        repr(candidate.text),
    )
```

运行：

```bash
python offline_demo.py
```

### 5.1 应观察什么

不要只检查“有文字输出”。至少确认：

- `vllm.__version__` 是 0.27.1；
- 三个请求都完成；
- Prompt token 数包含 Chat Template 引入的角色 token；
- `temperature=0` 下重复运行内容稳定；
- 首次运行包含下载、加载、编译与 warmup，不能当稳态延迟；
- 第二次进程内调用通常比模型初始化更短。

### 5.2 一个容易犯的错

`llm.generate(["你好"])` 不会自动把裸字符串变成目标模型的对话模板。Chat 模型优先使用 `llm.chat()`；若手动模板化，应验证：

```python
rendered = llm.get_tokenizer().apply_chat_template(
    messages[0],
    tokenize=False,
    add_generation_prompt=True,
)
print(repr(rendered))
```

“输出乱码”经常不是 GPU 或量化问题，而是输入协议不符合训练时模板。

## 6. 实验二：OpenAI 兼容在线服务

### 6.1 启动

```bash
vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --generation-config vllm \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85
```

本文刻意只监听 `127.0.0.1`。直接监听公网且没有网关鉴权，不是生产部署。

等待日志出现服务就绪后，在另一终端检查：

```bash
curl -f "http://127.0.0.1:${PORT}/health"
curl -s "http://127.0.0.1:${PORT}/v1/models"
```

### 6.2 非流式请求

```bash
curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"解释 PagedAttention\"}],
    \"temperature\": 0,
    \"max_tokens\": 64,
    \"seed\": 7
  }"
```

检查响应中的：

```text
choices[0].message.content
choices[0].finish_reason
usage.prompt_tokens
usage.completion_tokens
```

`finish_reason=length` 表示碰到 `max_tokens`，不等于模型自然回答完整。

### 6.3 流式请求与 TTFT

```bash
curl -N "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"逐步解释连续批处理\"}],
    \"temperature\": 0,
    \"max_tokens\": 128,
    \"stream\": true
  }"
```

`-N` 禁用 curl 输出缓冲。否则服务端已经流式发送，终端仍可能成批显示，让人误判 ITL。

### 6.4 Python 客户端

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
    timeout=60,
)

with client.chat.completions.stream(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "解释 KV block。"}],
    temperature=0,
    max_tokens=64,
) as stream:
    for event in stream:
        if event.type == "content.delta":
            print(event.delta, end="", flush=True)
```

客户端 API 会随 `openai` 包版本变化。遇到方法不存在时先记录客户端版本，不要先改服务端。

## 7. 从源码看最短调用链

固定 v0.27.1 后，一次在线 Chat 请求的主链不是“HTTP 直接调用 model.forward”：

```text
vllm/entrypoints/openai/chat_completion/api_router.py
  create_chat_completion()
    ↓
vllm/entrypoints/openai/chat_completion/serving.py
  OpenAIServingChat.create_chat_completion()
    ↓
vllm/v1/engine/async_llm.py
  AsyncLLM.generate()
  AsyncLLM.add_request()
    ↓
vllm/v1/engine/input_processor.py
  InputProcessor.process_inputs[_async]()
    ↓
vllm/v1/engine/__init__.py
  EngineCoreRequest
    ↓
vllm/v1/engine/core_client.py
  AsyncMPClient.add_request_async()
    ↓ ZMQ + msgpack
vllm/v1/engine/core.py
  EngineCoreProc.preprocess_add_request()
  EngineCore.step()
    ↓
Scheduler.schedule()
Executor.execute_model()
Scheduler.update_from_output()
```

关键对象 `EngineCoreRequest` 在 v0.27.1 中包含：

```text
request_id
prompt_token_ids / prompt_embeds
mm_features
sampling_params / pooling_params
arrival_time
priority
cache_salt
client_index
```

可追踪伪代码：

```python
async def online_request(http_json):
    rendered = render_chat(http_json["messages"])
    core_req = await input_processor.process_inputs_async(
        request_id=external_id,
        prompt=rendered,
        params=sampling_params,
    )
    input_processor.assign_request_id(core_req)
    output_queue = RequestOutputCollector(...)
    output_processor.add_request(core_req, output_queue)
    await async_mp_client.add_request_async(core_req)

    while True:
        output = await output_queue.get()
        yield output
        if output.finished:
            break
```

`assign_request_id()` 会保留外部 ID，同时默认追加随机后缀生成内部 ID，避免不同客户端重复 ID 导致状态串线。

## 8. 手算一次显存与张量形状

### 8.1 KV Cache 每 token

对普通 GQA 模型，单 token KV Cache 粗算：

$$
M_{\text{KV/token}}
=2\times L\times H_{\text{kv}}\times D_{\text{head}}\times S
$$

假设：

```text
层数 L = 24
KV heads = 4
head_dim = 128
BF16 = 2 bytes
```

则：

$$
2\times24\times4\times128\times2
=49,152\text{ bytes}
=48\text{ KiB/token}
$$

一个 2,048-token 请求约需：

$$
48\text{ KiB}\times2048=96\text{ MiB}
$$

若有 80 个同长度请求，仅理论 KV 已约 7.5 GiB。实际还要考虑 block 对齐、混合 Attention、临时张量和框架开销。

### 8.2 Attention 张量维度

本轮有 3 个请求，分别调度 `[1, 1, 126]` 个 token，总 query token 数：

$$
T=1+1+126=128
$$

若模型 hidden size 为 896，展平后的输入 hidden states 可看作：

```text
[T, hidden_size] = [128, 896]
```

GQA 的 K/V 投影每层概念形状为：

```text
K: [128, 4, 128]
V: [128, 4, 128]
```

物理 KV Cache 不必按三个请求连续排列；block table 把每个逻辑位置映射到物理 block。

## 9. 实验三：观察连续批处理

使用固定随机长度负载：

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --random-range-ratio 0.0 \
  --num-prompts 200 \
  --request-rate 8 \
  --max-concurrency 32 \
  --num-warmups 10 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --save-result \
  --save-detailed
```

再改成 burst：

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 200 \
  --request-rate inf \
  --max-concurrency 32 \
  --num-warmups 10
```

两轮不能只比较总吞吐。`request-rate=inf` 会在零时刻发出全部请求，排队形态与在线 Poisson 到达不同。

并行观察：

```bash
watch -n 1 nvidia-smi
curl -s "http://127.0.0.1:${PORT}/metrics" > metrics-after.txt
```

预期看到高并发时 active batch 与 KV 使用上升；请求完成后资源回落。GPU 利用率不一定恒定 100%，短模型还可能受 CPU、调度和 Kernel launch 限制。

## 10. 实验四：验证 Prefix Cache

构造 1,024-token 固定前缀与短随机后缀：

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "$MODEL" \
  --dataset-name random \
  --random-prefix-len 1024 \
  --random-input-len 64 \
  --random-output-len 32 \
  --num-prompts 100 \
  --request-rate 4 \
  --max-concurrency 16 \
  --save-result
```

注意这里总输入长度约为固定前缀 1,024 加随机上下文 64，不是 1,024。

先冷启动跑一轮，再不重启服务跑相同前缀。保存两轮 `/metrics`：

```bash
curl -s "http://127.0.0.1:${PORT}/metrics" > metrics-cold.txt
# 再跑同一负载
curl -s "http://127.0.0.1:${PORT}/metrics" > metrics-warm.txt
```

v0.27.1 可重点核对：

```text
vllm:prefix_cache_queries
vllm:prefix_cache_hits
vllm:time_to_first_token_seconds
```

命中率：

$$
hit\_rate=
\frac{\Delta prefix\_cache\_hits}
{\Delta prefix\_cache\_queries}
$$

计数器是累计值，应使用实验前后差值。全量 Prompt 命中时仍需重新计算末尾位置以取得 logits，因此 warm TTFT 不会变成零。

## 11. 指标应该怎样解释

端到端延迟可分解为：

$$
T_{\text{e2e}}
=T_{\text{HTTP}}
+T_{\text{tokenize}}
+T_{\text{queue}}
+T_{\text{prefill}}
+T_{\text{decode}}
+T_{\text{detokenize/network}}
$$

TTFT 主要覆盖首 token 前的路径；TPOT 是首 token 后的请求级平均；ITL 是相邻流式 token 的间隔分布。

诊断时建立对应关系：

```text
TTFT 高、queue 高        → 过载或并发上限
TTFT 高、cache hit 低    → 重复前缀没有复用
ITL 高、长 Prompt 同时来 → Prefill 干扰 Decode
吞吐低、GPU 低           → CPU/Tokenizer/请求不足/shape 开销
KV usage 高、preemption 增 → 容量不足或准入过宽
```

至少使用 P50/P95/P99。均值会掩盖少数超长 Prompt 对交互用户的影响。

## 12. 失败模式与分层排障

### 12.1 安装阶段

**现象：** `import vllm` 时缺少 CUDA 库。

先检查：

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
ldconfig -p | grep -E 'libcuda|libcudart' || true
```

容器能看到 `nvidia-smi`，不保证动态库和 PyTorch wheel 一定匹配。

### 12.2 模型加载阶段

**现象：** 401/403 或找不到文件。

检查模型许可、token、revision、代理与缓存目录。`--trust-remote-code` 会执行模型仓库代码，只应对审查过的 revision 使用。

### 12.3 启动 OOM

先判断 OOM 在哪里：

```text
加载权重时 → 权重本身装不下，考虑量化或并行
profile/warmup 时 → 临时峰值、编译或 CUDA Graph
KV 初始化时 → max_model_len / KV 预算不成立
```

不要无脑降低 `gpu-memory-utilization`。它会减少当前实例预算，通常也减少 KV 容量。权重装不下时，降低它可能让启动更早失败。

用于定位的对照：

```bash
vllm serve "$MODEL" \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.80 \
  --enforce-eager
```

若 eager 可启动而默认模式失败，应继续检查编译与 CUDA Graph 的额外显存，不要把 eager 直接当最终性能配置。

### 12.4 请求返回 400

保存完整响应体：

```bash
curl -sS -D response.headers -o response.json \
  "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"wrong-name","messages":[]}'
```

常见原因是 served model name、Chat 请求结构、总长度或不支持的采样字段，不是 EngineCore 崩溃。

### 12.5 流式输出“攒一坨”

依次排除：

1. curl 是否使用 `-N`；
2. Python 是否 `flush=True`；
3. 反向代理是否开启响应缓冲；
4. 客户端 SDK 是否聚合 delta；
5. 服务端 ITL 是否真的升高。

### 12.6 运行一段时间后变慢

同时采集：

```bash
curl -s "http://127.0.0.1:${PORT}/metrics" > slow.metrics
nvidia-smi dmon -s pucvmet -d 1 -c 60 > slow.gpu.txt
```

优先检查请求长度分布变化、KV 使用率、preemption、Prefix Cache churn、GPU 降频和 CPU 饱和。不要只重启服务掩盖根因。

## 13. 第一轮参数取舍

### `max_model_len`

它是产品能力边界。缩短它降低最坏 KV 压力，但会拒绝更长上下文。

### `gpu_memory_utilization`

v0.27.1 `CacheConfig` 默认是 0.92。它是当前实例模型执行器的显存预算比例，不是“KV Cache 占总显存比例”。

### `max_num_seqs`

v0.27.1 `SchedulerConfig` 类默认 128；真实启动值仍应以最终配置日志为准。提高它可能增加 Decode 并发，也会增加 KV 与 batch 状态压力。

### `max_num_batched_tokens`

限制单轮计划 token 总数。大值倾向于提升 Prefill 吞吐，小值通常更有利于交互 Decode 的 ITL，但结论必须在目标负载上验证。

第一轮只改一个变量。若同时改模型长度、显存比例和 token budget，结果无法归因。

## 14. 自检

- [ ] 能解释“能生成”与“能在线服务”的差别。
- [ ] 能区分 continuous batching 与分页 KV Cache。
- [ ] 能在日志中确认 vLLM、模型 revision 和最终启动参数。
- [ ] 能用 `LLM.chat()` 跑通确定性离线实验。
- [ ] 能用 curl 与 OpenAI 客户端分别验证在线服务。
- [ ] 能画出 API → AsyncLLM → EngineCore 的最短源码链。
- [ ] 能说出 `EngineCoreRequest` 至少五个关键字段。
- [ ] 能按层数、KV head、head dim 和 dtype 手算 KV/token。
- [ ] 能用 `vllm bench serve` 区分稳态到达与突发到达。
- [ ] 能用累计 counter 的差值计算 Prefix Cache 命中率。
- [ ] 能区分加载 OOM、warmup OOM 与运行时 KV 压力。
- [ ] 能解释为什么降低 `gpu_memory_utilization` 不一定解决权重 OOM。
- [ ] 能同时观察 TTFT、ITL、吞吐、KV usage 与 preemption。

## 参考资料

- [vLLM v0.27.1 源码](https://github.com/vllm-project/vllm/tree/v0.27.1)
- [vLLM Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart.html)
- [OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html)
- [vLLM Benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli.html)
- [vLLM Metrics](https://docs.vllm.ai/en/stable/design/metrics.html)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
