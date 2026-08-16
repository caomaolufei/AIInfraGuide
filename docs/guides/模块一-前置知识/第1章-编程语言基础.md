---
title: "第1章：编程语言基础"
description: "掌握 AI Infra 开发所需的 Python 进阶、C/C++ 核心和 Linux 开发环境，为后续 CUDA 编程和系统开发打下坚实基础"
pubDate: 2026-04-16
category: "prerequisites"
order: 1
tags: ["Python", "C++", "Linux", "开发环境", "前置知识"]
---

AI Infra 处在算法、框架、编译器、操作系统和硬件的交界处。同一个问题往往要在多种语言和工具之间来回切换：用 Python 描述模型与实验，用 C++ 实现性能敏感路径，用 CUDA 驱动 GPU，再用 Linux 工具定位进程、内存、动态库和网络问题。

本章不追求把每门语言的语法完整讲一遍，而是建立一条够用的主线：**能读懂、能修改、能调试，也知道性能成本发生在哪里**。

<!-- more -->

## 📑 目录

- [1. 学习目标与开发全景](#1-学习目标与开发全景)
- [2. Python：从脚本到工程](#2-python从脚本到工程)
- [3. Python 并发与性能分析](#3-python-并发与性能分析)
- [4. C/C++：内存、生命周期与资源管理](#4-cc内存生命周期与资源管理)
- [5. C++ 编译、链接、模板与构建系统](#5-c-编译链接模板与构建系统)
- [6. Python 与 C++ 互操作](#6-python-与-c-互操作)
- [7. Linux 开发基本功](#7-linux-开发基本功)
- [8. 环境隔离与 GPU 软件栈](#8-环境隔离与-gpu-软件栈)
- [9. Git 协作与问题定位方法](#9-git-协作与问题定位方法)
- [10. 综合实战：给 Python 添加一个 C++ 算子](#10-综合实战给-python-添加一个-c-算子)
- [总结](#总结)
- [自我检验清单](#自我检验清单)
- [练习题](#练习题)
- [参考资料](#参考资料)

---

## 1. 学习目标与开发全景

完成本章后，你应该能够：

1. 解释 Python 名字、对象、引用、可变性和作用域之间的关系；
2. 熟练使用迭代器、生成器、装饰器、上下文管理器与类型标注；
3. 根据任务性质选择线程、进程或异步 I/O，并用工具找到 Python 热点；
4. 读懂 C/C++ 中的指针、引用、对象生命周期、RAII 和智能指针；
5. 解释一个 C++ 源文件从预处理到动态装载的全过程；
6. 看懂常见的模板、STL 容器和 CMake 项目；
7. 用 `pybind11` 理解 Python 调用 C++ 扩展的基本边界；
8. 在 Linux 上管理文件、进程、权限、环境变量、动态库和远程任务；
9. 区分 GPU 驱动、CUDA Toolkit、CUDA Runtime 和框架自带 CUDA 组件；
10. 用可复现的步骤定位“代码错、环境错还是性能差”。

### 1.1 三层语言各自负责什么

| 层次 | 常用技术 | 优势 | 典型工作 |
|------|----------|------|----------|
| 编排层 | Python、Shell | 开发快、生态丰富 | 模型定义、训练脚本、任务编排、数据处理 |
| 系统层 | C/C++ | 可控的内存与性能、易连接底层 API | 框架核心、算子调度、通信库、Python 扩展 |
| 设备层 | CUDA C++、Triton | 显式表达 GPU 并行 | Kernel、算子融合、访存优化 |

三层不是互相替代的关系。一个 `torch.matmul(x, w)` 看起来只是一行 Python，实际可能经历：

```text
Python API
  └─ 算子分发器（C++）
      └─ cuBLAS / 自定义 CUDA Kernel
          └─ GPU 指令与显存访问
```

因此，AI Infra 工程师最重要的语言能力不是“背语法”，而是**沿调用链跨层追踪问题**。

### 1.2 开始前的最小工具箱

以下命令不要求现在全部掌握，但应该知道它们分别回答什么问题：

```bash
python --version       # 正在使用哪个 Python
which python           # 这个可执行文件来自哪里
python -m pip --version

g++ --version          # C++ 编译器版本
cmake --version        # 构建系统版本

uname -a               # 内核与系统信息
ps aux                 # 有哪些进程
free -h                # 主机内存使用
df -h                   # 文件系统容量

nvidia-smi             # 驱动可见的 GPU 与进程
nvcc --version         # CUDA Toolkit 中的编译器版本
```

> **先建立诊断习惯**：环境问题出现时，先记录命令、输出、工作目录和版本，不要立刻反复重装。可观察的信息越完整，定位越快。

---

## 2. Python：从脚本到工程

### 2.1 名字、对象与引用

Python 变量更准确的说法是“**名字绑定到对象**”。赋值通常不会复制对象，只会建立新的引用：

```python
a = [1, 2]
b = a
b.append(3)

print(a)        # [1, 2, 3]
print(a is b)   # True：指向同一个对象
print(a == b)   # True：值相等
```

需要区分两个运算：

- `is` 比较对象身份，适合写 `x is None`；
- `==` 调用值比较逻辑，比较“内容是否相等”。

函数参数同样是名字绑定。函数收到的是对象引用，不是“按引用传递”或“按值传递”这两个传统标签中的任意一个：

```python
def mutate(xs: list[int]) -> None:
    xs.append(1)          # 修改调用方可见的同一个列表

def rebind(xs: list[int]) -> None:
    xs = [1,2]              # 只让局部名字 xs 指向新列表

data: list[int] = []
mutate(data)
print(data)               # [1]

rebind(data)
print(data)               # 仍然是 [1]
```

这套模型能解释大量常见 Bug：默认可变参数、浅拷贝、缓存污染以及多线程共享状态。

### 2.2 可变对象、不可变对象与拷贝

常见不可变对象包括 `int`、`float`、`bool`、`str`、`bytes` 和元素均不可变时的 `tuple`；常见可变对象包括 `list`、`dict`、`set` 和大多数自定义实例。

#### 默认参数只创建一次

```python
# 错误：同一个列表会被多次调用共享
def collect_bad(x: int, result: list[int] = []) -> list[int]:
    result.append(x)
    return result

# 正确：用 None 表示“未提供”
def collect(x: int, result: list[int] | None = None) -> list[int]:
    if result is None:
        result = []
    result.append(x)
    return result
```

默认参数在函数定义时求值，而不是每次调用时求值。

#### 浅拷贝与深拷贝

```python
import copy

src = [[1], [2]]
shallow = src.copy()           # 只复制最外层列表
deep = copy.deepcopy(src)      # 递归复制内部对象

shallow[0].append(9)
print(src)                     # [[1, 9], [2]]
print(deep)                    # [[1], [2]]
```

深拷贝并非默认答案。对大型张量或模型状态做无意复制可能产生巨大的内存和时间开销。更好的做法是先明确：哪些数据需要独占，哪些可以只读共享。

### 2.3 作用域、闭包与 LEGB

Python 按 **LEGB** 顺序查找名字：Local（局部）→ Enclosing（外层函数）→ Global（模块）→ Builtins（内置）。

闭包会保存外层作用域中的变量：

```python
from collections.abc import Callable

def make_multiplier(scale: float) -> Callable[[float], float]:
    def multiply(x: float) -> float:
        return x * scale
    return multiply

double = make_multiplier(2.0)
print(double(3.0))  # 6.0
```

循环中创建闭包时要注意**延迟绑定**：闭包在调用时才读取外层名字。

```python
# 三个函数最终都读取同一个 i，结果都是 2
bad = [lambda: i for i in range(3)]

# 用默认参数在每轮定义时保存当前值
good = [lambda i=i: i for i in range(3)]
print([fn() for fn in good])  # [0, 1, 2]
```

如果需要在内层函数中重新绑定外层函数的名字，使用 `nonlocal`；重新绑定模块全局名字则使用 `global`。优先通过参数和返回值传递状态，通常更易测试。

### 2.4 类、数据模型与协议

Python 的特殊方法让对象接入语言协议。例如，实现 `__len__` 后可调用 `len(obj)`，实现 `__iter__` 后可被 `for` 遍历。

```python
from dataclasses import dataclass
from collections.abc import Iterator

@dataclass
class Batch:
    samples: list[list[float]]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[list[float]]:
        return iter(self.samples)

batch = Batch([[1.0, 2.0], [3.0, 4.0]])
print(len(batch))
for sample in batch:
    print(sample)
```

常见特殊方法及用途：

| 方法 | 触发方式 | 典型用途 |
|------|----------|----------|
| `__init__` | 创建实例后 | 初始化实例状态 |
| `__repr__` | `repr(obj)`、调试器 | 无歧义的调试表示 |
| `__call__` | `obj(...)` | 让实例表现得像函数 |
| `__enter__` / `__exit__` | `with` | 管理资源生命周期 |
| `__iter__` / `__next__` | `for`、`next` | 定义迭代协议 |
| `__getitem__` | `obj[key]` | 索引、切片或映射访问 |

#### 实例方法、类方法和静态方法

- 实例方法的第一个参数是 `self`，操作具体对象；
- `@classmethod` 的第一个参数是 `cls`，常用于备选构造器；
- `@staticmethod` 不接收隐式对象，仅用于把相关工具函数组织在类命名空间中。

```python
@dataclass
class Device:
    kind: str
    index: int

    @classmethod
    def parse(cls, value: str) -> "Device":
        kind, index = value.split(":")
        return cls(kind=kind, index=int(index))

device = Device.parse("cuda:0")
```

### 2.5 迭代器与生成器：按需生产数据

**可迭代对象**能返回迭代器；**迭代器**保存遍历状态，并通过 `__next__` 逐个返回元素。生成器函数是编写迭代器的简洁方式：

```python
from collections.abc import Iterator

def read_batches(path: str, batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            batch.append(line.rstrip("\n"))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch
```

`yield` 返回一个值后暂停函数，下一次迭代从暂停位置继续。与一次性构造完整列表相比，生成器的峰值内存通常只与一个批次相关，适合流式数据管线。

需要记住：

- 生成器通常只能消费一次；
- 惰性求值会把异常推迟到迭代时；
- 把生成器转成 `list` 会重新物化所有元素，失去节省内存的优势。

### 2.6 装饰器：在调用边界附加行为

装饰器接收一个可调用对象，并返回另一个可调用对象。下面的计时器保留了原函数的元信息：

```python
from collections.abc import Callable
from functools import wraps
from time import perf_counter
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def timed(fn: Callable[P, R]) -> Callable[P, R]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (perf_counter() - start) * 1_000
            print(f"{fn.__name__}: {elapsed_ms:.3f} ms")
    return wrapper

@timed
def preprocess(values: list[float]) -> list[float]:
    return [value * 2 for value in values]
```

`@timed` 等价于 `preprocess = timed(preprocess)`。`functools.wraps` 会保留函数名、文档字符串等属性，这对日志、测试和框架反射很重要。

装饰器适合日志、缓存、权限检查和性能观测；如果只是单个调用点需要计时，一个普通的 `with timer():` 往往更直白，不必为了“高级”而使用装饰器。

### 2.7 上下文管理器：让资源必定释放

`with` 把资源的获取和释放放在同一个词法范围内，即使发生异常，也会执行清理逻辑：

```python
from contextlib import contextmanager
from collections.abc import Iterator
from time import perf_counter

@contextmanager
def timer(name: str) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - start) * 1_000
        print(f"{name}: {elapsed_ms:.3f} ms")

with timer("load data"):
    data = [value for value in range(100_000)]
```

文件、锁、数据库连接、临时目录、自动混合精度上下文都适合这种模式。它和 C++ 的 RAII 解决的是同一个问题：**资源生命周期必须有清晰边界**。

### 2.8 异常处理：只捕获能够处理的错误

```python
def parse_world_size(raw: str) -> int:
    try:
        world_size = int(raw)
    except ValueError as exc:
        raise ValueError(f"WORLD_SIZE 必须是整数，实际为 {raw!r}") from exc

    if world_size <= 0:
        raise ValueError("WORLD_SIZE 必须大于 0")
    return world_size
```

建议遵循三条原则：

1. 捕获具体异常，不要用空的 `except:` 吞掉 `KeyboardInterrupt` 等信号；
2. 只有在能恢复、补充上下文或转换抽象层时才捕获；
3. 用 `raise ... from exc` 保留根因链。

不要把异常用于正常的高频控制流，尤其是在性能敏感循环中。

### 2.9 类型标注与数据类

类型标注主要服务于静态检查、IDE 和读者，不会自动在运行时验证类型：

```python
from dataclasses import dataclass
from collections.abc import Sequence

@dataclass(frozen=True)
class ShardSpec:
    rank: int
    world_size: int

def shard(values: Sequence[int], spec: ShardSpec) -> Sequence[int]:
    return values[spec.rank::spec.world_size]
```

常用选择：

- 参数只要求“可迭代”或“序列”能力时，标注 `Iterable[T]`、`Sequence[T]` 等抽象协议；
- 返回具体容器时可标注 `list[T]`、`dict[K, V]`；
- `T | None` 表示值可能缺失；
- `Protocol` 可描述结构化接口，减少不必要的继承；
- `dataclass` 适合主要承载数据、规则简单的对象。

类型系统的目标是尽早暴露接口误用，不是把动态语言硬写成 Java。公共边界优先标清，局部显而易见的变量不必逐个标注。

---

## 3. Python 并发与性能分析

### 3.1 先分清并发与并行

- **并发（concurrency）**：多个任务在同一时间段内推进；
- **并行（parallelism）**：多个任务在同一时刻真正执行。

选择并发模型前，先判断瓶颈：

| 任务类型 | 典型例子 | 常见选择 |
|----------|----------|----------|
| I/O 密集 | 请求服务、读取大量小文件 | 线程或 `asyncio` |
| Python CPU 密集 | 纯 Python 解析、复杂循环 | 多进程或原生扩展 |
| 原生算子密集 | NumPy/PyTorch/CUDA 运算 | 由底层线程池/GPU 并行，Python 负责调度 |
| 大数据跨进程 | 数据加载、共享缓存 | 多进程 + 共享内存，谨慎序列化 |

### 3.2 GIL 到底限制了什么

常见 CPython 构建使用全局解释器锁（GIL）保护解释器内部状态。同一进程中，通常只有一个线程能同时执行 Python 字节码。因此，两个执行纯 Python CPU 循环的线程一般不能获得两倍速度。

但这不等于“Python 线程没有用”：

- 阻塞 I/O 时解释器会释放 GIL；
- NumPy、PyTorch 等原生扩展可在耗时计算时释放 GIL；
- 后台日志、预取和轻量控制任务常适合线程。

GIL 是 CPython 实现细节，不应泛化成所有 Python 实现或所有构建的永恒规则。判断性能时要测量当前解释器和依赖，而不是只背结论。

### 3.3 线程：共享内存，也共享风险

```python
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

def fetch_size(url: str) -> int:
    with urlopen(url, timeout=10) as response:
        return len(response.read())

urls = ["https://example.com"] * 4
with ThreadPoolExecutor(max_workers=4) as pool:
    sizes = list(pool.map(fetch_size, urls))
```

线程共享进程内存，传数据便宜，但对同一可变对象的“读取—修改—写回”可能产生竞态。需要锁时让临界区尽可能小：

```python
from threading import Lock

lock = Lock()
counter = 0

def increment() -> None:
    global counter
    with lock:
        counter += 1
```

不要依赖“某个内置操作看起来是原子的”来设计正确性；解释器版本、对象实现或复合操作都可能破坏这种假设。

### 3.4 多进程：绕开 GIL，但数据搬运有成本

```python
from concurrent.futures import ProcessPoolExecutor

def count_primes(limit: int) -> int:
    def is_prime(value: int) -> bool:
        if value < 2:
            return False
        return all(value % divisor for divisor in range(2, int(value**0.5) + 1))
    return sum(is_prime(value) for value in range(limit))

if __name__ == "__main__":
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(count_primes, [50_000] * 4))
```

`if __name__ == "__main__":` 对使用 `spawn` 启动子进程的平台尤其重要，可避免子进程重新导入模块时递归创建新进程。

多进程的主要成本包括：

- 创建进程和独立解释器；
- 参数与返回值的序列化/反序列化；
- 复制或映射内存；
- 进程间通信与同步。

如果每个任务只做几十微秒工作，调度成本可能比计算更高。应增大任务粒度，并尽量避免来回传递大型数组或张量。

> **GPU 关联**：CUDA 上下文和多进程启动方式组合不当会引发初始化错误或额外显存占用。使用框架的数据加载与分布式能力时，应遵循框架推荐的启动器和 start method，不要在父进程初始化 CUDA 后随意 `fork`。

### 3.5 `asyncio`：协作式 I/O 并发

`asyncio` 用事件循环调度协程。协程只会在 `await` 处主动让出控制权：

```python
import asyncio

async def worker(name: str, delay: float) -> str:
    await asyncio.sleep(delay)  # 模拟非阻塞 I/O
    return name

async def main() -> None:
    results = await asyncio.gather(
        worker("a", 0.2),
        worker("b", 0.1),
    )
    print(results)

asyncio.run(main())
```

在事件循环中直接执行阻塞 I/O 或长时间 CPU 计算会卡住所有协程。此时应改用异步库，或把阻塞工作放到线程/进程执行器。

### 3.6 正确做基准测试

性能测试最常见的错误是只测一次、把初始化算进去，或者忘记 GPU 异步执行。

纯 CPU 小片段可用 `timeit`：

```bash
python -m timeit -s 'xs = list(range(1000))' 'sum(xs)'
```

手工测试至少要做到：

```python
from statistics import median
from time import perf_counter

def benchmark(fn, warmup: int = 5, repeat: int = 20) -> float:
    for _ in range(warmup):
        fn()

    samples = []
    for _ in range(repeat):
        start = perf_counter()
        fn()
        samples.append(perf_counter() - start)
    return median(samples)
```

对于 GPU，Kernel 启动通常是异步的。CPU 计时器只量到“提交任务”的时间，需要在边界同步，或使用框架提供的 GPU Event。同步会改变流水线行为，因此端到端吞吐和单算子延迟应分别测量。

### 3.7 从“慢”定位到具体代码

建议按由粗到细的顺序：

1. **端到端计时**：确认回归真实存在；
2. **阶段计时**：区分数据、计算、通信和保存；
3. **函数级 profiling**：用 `cProfile` 找累计耗时高的函数；
4. **行级/采样 profiling**：进一步定位热点行或原生调用栈；
5. **内存 profiling**：用 `tracemalloc` 等工具找 Python 分配增长；
6. **GPU profiling**：进入 PyTorch Profiler、Nsight Systems/Compute 等专用工具。

```bash
python -m cProfile -o profile.out train.py
python -m pstats profile.out
```

在 `pstats` 中可按累计时间排序：

```text
sort cumulative
stats 30
```

检查 Python 内存分配：

```python
import tracemalloc

tracemalloc.start()
# 运行待检查代码
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics("lineno")[:10]:
    print(stat)
```

### 3.8 Python 性能优化的优先级

推荐顺序如下：

1. 先改算法复杂度和数据结构；
2. 减少不必要的 I/O、复制和序列化；
3. 把细粒度 Python 循环改成批处理/向量化原生算子；
4. 减少 Python 与 C++/GPU 边界的频繁往返；
5. 确认热点后，再写 C++、CUDA 或 Triton 扩展。

例如，一百万次小算子调用即使每次都很快，也会累积大量调度开销。把数据组成批次后调用一次大算子，通常比微调 Python 语法有效得多。

---

## 4. C/C++：内存、生命周期与资源管理

### 4.1 值、地址、指针与引用

```cpp
#include <iostream>

int main() {
    int value = 42;
    int* ptr = &value;  // ptr 保存 value 的地址
    int& ref = value;   // ref 是 value 的别名

    *ptr = 7;           // 解引用指针并写入
    ref += 1;

    std::cout << value << '\n';  // 8
}
```

可以先用这张表建立直觉：

| 概念 | 能否为空 | 能否改指向 | 常见用途 |
|------|----------|------------|----------|
| 值 `T` | 否 | 不适用 | 拥有一个独立对象 |
| 指针 `T*` | 可以是 `nullptr` | 可以 | 可选对象、数组、底层接口、非拥有观察 |
| 引用 `T&` | 语言语义上必须绑定对象 | 不可以重新绑定 | 必须存在的别名、函数参数 |
| 常量引用 `const T&` | 必须绑定对象 | 不可以 | 避免复制且不允许修改 |

函数参数表达的接口意图应尽可能清晰：

```cpp
void consume(T value);          // 按值：函数获得自己的对象
void update(T& value);          // 可变借用：函数会修改调用方对象
void inspect(const T& value);   // 只读借用：避免复制
void optional(T* value);        // 指针可能为空，通常不拥有对象
```

对小型标量（`int`、`float`、指针）按值传递通常最清楚；不要机械地把所有参数都改成引用。

### 4.2 数组、指针与连续内存

原生数组在大多数表达式中会退化为首元素指针，因此单独的 `T*` 不携带长度：

```cpp
void scale(float* data, std::size_t size, float factor) {
    for (std::size_t i = 0; i < size; ++i) {
        data[i] *= factor;
    }
}
```

现代 C++ 中优先用能表达长度的类型：

```cpp
#include <span>

void scale(std::span<float> values, float factor) {
    for (float& value : values) {
        value *= factor;
    }
}
```

`std::span` 不拥有数据，只是“指针 + 长度”的轻量视图。调用期间底层内存必须仍然有效。

二维矩阵通常线性存放。行主序矩阵 `A[M][N]` 的 `(row, col)` 对应偏移：

$$
\text{offset} = row \times N + col
$$

这个公式会在 CUDA 中反复出现。访问顺序是否连续，直接影响 CPU cache 和 GPU 合并访存效率。

### 4.3 对象存储期与生命周期

常见存储位置的说法是“栈”和“堆”，但更准确地应关注**对象的存储期和所有权**：

```cpp
void example() {
    int local = 1;                  // 自动存储期，离开作用域销毁
    auto buffer = new float[1024];  // 动态分配，必须显式 delete[]
    delete[] buffer;
}
```

手工 `new`/`delete` 容易产生：

- 内存泄漏：忘记释放；
- 重复释放：同一地址释放两次；
- 悬空指针：对象已销毁，指针仍被使用；
- 分配/释放形式不匹配：`new[]` 配了 `delete`；
- 异常路径泄漏：释放前函数提前退出。

因此现代 C++ 的默认原则是：**让资源由对象管理，让对象由作用域管理。**

### 4.4 RAII：把资源绑定到对象生命周期

RAII（Resource Acquisition Is Initialization）在构造时获取资源，在析构时释放资源：

```cpp
#include <cstdio>
#include <stdexcept>

class File {
public:
    explicit File(const char* path)
        : handle_(std::fopen(path, "rb")) {
        if (handle_ == nullptr) {
            throw std::runtime_error("failed to open file");
        }
    }

    ~File() {
        std::fclose(handle_);
    }

    File(const File&) = delete;
    File& operator=(const File&) = delete;

private:
    std::FILE* handle_;
};
```

无论函数正常返回还是抛出异常，局部对象的析构函数都会执行。文件、锁、主机内存、GPU 内存、CUDA Stream 都可以用同一思想管理。

实际项目中优先使用标准库已有的 RAII 类型，如 `std::vector`、`std::string`、`std::fstream`、`std::lock_guard` 和智能指针，不要重复造轮子。

### 4.5 智能指针与所有权

```cpp
#include <memory>

auto unique = std::make_unique<Buffer>(1024);  // 唯一所有权
auto shared = std::make_shared<Buffer>(1024);  // 共享所有权
```

| 类型 | 语义 | 成本与注意点 |
|------|------|--------------|
| `std::unique_ptr<T>` | 唯一拥有，可移动不可复制 | 成本接近裸指针，默认首选 |
| `std::shared_ptr<T>` | 引用计数共享拥有 | 有计数与控制块成本，循环引用需 `weak_ptr` |
| `std::weak_ptr<T>` | 不增加引用计数的观察者 | 使用前要 `lock()` 检查对象是否仍存在 |
| `T*` / `T&` | 通常表示非拥有访问 | 被指向对象必须活得足够久 |

如果所有权没有共享需求，不要使用 `shared_ptr`。“避免思考所有权”不是共享所有权的合理理由。

### 4.6 拷贝、移动与 Rule of Zero

`std::vector` 等容器拥有资源，复制会复制内容，移动则转移内部资源：

```cpp
#include <utility>
#include <vector>

std::vector<float> make_buffer() {
    std::vector<float> values(1'000'000, 0.0f);
    return values;  // 编译器通常消除复制，必要时可移动
}

auto a = make_buffer();
auto b = std::move(a);  // a 仍有效，但内容处于未指定状态
```

如果类只由标准库 RAII 成员组成，通常无需手写析构、复制和移动操作，这就是 **Rule of Zero**。只有直接管理特殊资源时才需要自定义这些操作，并清晰定义所有权。

不要随手对返回值写 `std::move`；它可能阻止返回值优化。也不要在移动后假设源对象仍保留原内容。

### 4.7 `const`：把不可修改写进接口

```cpp
const float* p1;        // 指向 const float：不能通过 p1 改值
float* const p2 = ptr;  // const 指针：不能让 p2 改指向
const float* const p3 = ptr;
```

成员函数末尾的 `const` 表示不修改对象的可观察状态：

```cpp
class TensorView {
public:
    std::size_t size() const { return size_; }
private:
    std::size_t size_{};
};
```

`const` 有助于编译器和读者验证接口，但不能自动解决并发安全与底层别名问题。

### 4.8 未定义行为为什么危险

越界访问、使用悬空引用、有符号整数溢出、数据竞争等可能触发未定义行为（Undefined Behavior，UB）。编译器可以假设 UB 永不发生，并据此优化；程序结果不一定只是“报错”，也可能在调试版正常、优化版悄悄算错。

```cpp
std::vector<int> values(4);
values[4] = 1;  // 越界，未定义行为
```

调试时可启用警告和 Sanitizer：

```bash
g++ -std=c++20 -Wall -Wextra -Wpedantic \
    -fsanitize=address,undefined -g main.cpp -o app
./app
```

AddressSanitizer 常用于发现越界、use-after-free；UndefinedBehaviorSanitizer 检查多类 UB；ThreadSanitizer 可定位数据竞争，但通常不能和 AddressSanitizer 同时启用。

---

## 5. C++ 编译、链接、模板与构建系统

### 5.1 从源代码到可执行文件

```text
源文件 .cpp
  └─ 预处理：展开 #include、宏和条件编译
      └─ 编译：语法/类型检查并生成汇编
          └─ 汇编：生成目标文件 .o
              └─ 链接：解析符号并生成可执行文件或库
```

可以逐步观察：

```bash
g++ -E main.cpp -o main.ii       # 只预处理
g++ -S main.ii -o main.s         # 生成汇编
g++ -c main.cpp -o main.o        # 生成目标文件
g++ main.o -o app                # 链接
```

头文件通常放声明，源文件放定义：

```cpp
// add.h
#pragma once
int add(int lhs, int rhs);
```

```cpp
// add.cpp
#include "add.h"
int add(int lhs, int rhs) {
    return lhs + rhs;
}
```

### 5.2 编译错误、链接错误与运行时装载错误

三类错误不要混在一起：

| 阶段 | 典型报错 | 常见原因 |
|------|----------|----------|
| 编译 | `not declared`、类型不匹配 | 缺声明、模板实例化失败、语法错误 |
| 链接 | `undefined reference` | 只声明未定义、漏链接库、ABI/签名不一致 |
| 运行时装载 | `cannot open shared object file` | 动态库不在搜索路径、架构不匹配 |

定位符号和依赖的常用命令：

```bash
nm -C build/app | less          # 查看并反修饰 C++ 符号
ldd build/app                   # Linux：查看动态库依赖
readelf -d build/app            # 查看 ELF 动态段
```

在 macOS 上，对应工具通常是 `otool -L`。

### 5.3 静态库与动态库

- 静态库（Linux 常见 `.a`）在链接时把所需代码并入目标文件；部署直接，但产物可能更大；
- 动态库（Linux `.so`、macOS `.dylib`）在装载/运行时解析；可被多个程序共享，但要处理搜索路径与 ABI 兼容。

Linux 动态库搜索会受到二进制的 RPATH/RUNPATH、`LD_LIBRARY_PATH`、系统缓存与默认目录影响。临时设置 `LD_LIBRARY_PATH` 适合诊断，不应成为掩盖安装布局问题的长期万能方案。

### 5.4 ABI：源码兼容不代表二进制兼容

ABI（Application Binary Interface）规定二进制层面的调用约定、名称修饰、对象布局和标准库类型实现等。以下情况都可能造成 ABI 不兼容：

- 编译器或标准库不一致；
- 编译选项改变对象布局或 C++ ABI；
- 库升级后导出符号发生变化；
- Debug/Release 运行库混用；
- CPU 架构不同。

Python C++ 扩展“能编译但导入失败”时，除了 Python 路径，还要检查扩展和依赖库的 ABI、架构与符号。

### 5.5 模板：在编译期生成类型特化代码

```cpp
template <typename T>
T add(T lhs, T rhs) {
    return lhs + rhs;
}

auto i = add(1, 2);        // 实例化 add<int>
auto f = add(1.0f, 2.0f);  // 实例化 add<float>
```

模板定义通常要放在头文件中，因为编译器在使用点需要看到完整定义才能实例化。CUDA 和高性能计算代码大量使用模板来表达：

- 数据类型：`float`、`half`、`bfloat16`；
- tile 大小和线程块配置；
- 布局与转置选项；
- 编译期分支和特化实现。

非类型模板参数可以把尺寸变成编译期常量：

```cpp
template <typename T, std::size_t N>
T sum(const std::array<T, N>& values) {
    T result{};
    for (const T& value : values) {
        result += value;
    }
    return result;
}
```

模板错误往往很长。定位时从最前面的“用户代码实例化位置”和最底层的真正约束失败开始看，不要被中间几十层展开吓住。

### 5.6 STL 容器与迭代器

| 类型 | 内存/访问特征 | 常见用途 |
|------|---------------|----------|
| `std::vector<T>` | 连续内存，随机访问 O(1) | 数值缓冲区，默认序列容器 |
| `std::array<T, N>` | 固定大小、连续内存 | 编译期已知的小数组 |
| `std::deque<T>` | 分段存储，两端插入高效 | 队列、滑动窗口 |
| `std::unordered_map<K,V>` | 哈希表，平均 O(1) 查找 | 无序键值索引 |
| `std::map<K,V>` | 有序树，O(log n) | 需要有序遍历/范围查询 |

`std::vector` 扩容可能重新分配内存，使原有指针、引用和迭代器失效：

```cpp
std::vector<int> values;
values.reserve(1024);  // 已知规模时可减少重复扩容
```

性能敏感代码除了看大 O 复杂度，还要看连续性、分配次数、缓存局部性和元素大小。很多场景中，线性扫描连续 `vector` 比理论查找复杂度更低但指针跳转频繁的结构更快。

### 5.7 最小 CMake 项目

目录：

```text
vector_ops/
├── CMakeLists.txt
├── include/vector_ops.h
└── src/vector_ops.cpp
```

`CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.20)
project(vector_ops LANGUAGES CXX)

add_library(vector_ops src/vector_ops.cpp)
target_include_directories(vector_ops PUBLIC include)
target_compile_features(vector_ops PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(vector_ops PRIVATE -Wall -Wextra -Wpedantic)
endif()
```

构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

现代 CMake 的核心是 **target**。用 `target_include_directories`、`target_link_libraries`、`target_compile_features` 描述依赖，少用全局的 `include_directories` 和手工拼接编译命令。

### 5.8 C++ 并发的最小认知

多个线程并发访问同一内存，其中至少一个写入且缺乏同步，就可能形成数据竞争，属于未定义行为。

```cpp
#include <mutex>

std::mutex mutex;
int counter = 0;

void increment() {
    std::lock_guard<std::mutex> guard(mutex);
    ++counter;
}
```

互斥锁保证临界区互斥，原子类型适合简单共享状态，条件变量用于等待状态变化。不要把 `volatile` 当作线程同步工具；它不提供原子性或线程间 happens-before 关系。

高性能并发优化前，首先保证正确性，并使用 ThreadSanitizer 等工具检查数据竞争。

---

## 6. Python 与 C++ 互操作

### 6.1 为什么需要跨语言边界

常见目的包括：

- 复用已有 C/C++ 库；
- 把 Python 热点循环下沉到原生代码；
- 接入驱动、通信库或自定义硬件接口；
- 为 PyTorch 等框架添加自定义算子。

跨语言并不自动变快。一次调用要做参数检查、类型/布局转换、可能的数据复制和异常翻译。如果每个元素都跨一次边界，绑定开销可能完全吞掉原生计算收益。理想接口通常传递一个批次或连续缓冲区。

### 6.2 `ctypes`：调用稳定的 C ABI

C++ 先导出一个 C 接口：

```cpp
// scale.cpp
extern "C" void scale(float* data, int size, float factor) {
    for (int i = 0; i < size; ++i) {
        data[i] *= factor;
    }
}
```

编译动态库：

```bash
g++ -O3 -shared -fPIC scale.cpp -o libscale.so
```

Python 侧调用：

```python
import ctypes
from array import array

library = ctypes.CDLL("./libscale.so")
library.scale.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_float,
]
library.scale.restype = None

values = array("f", [1.0, 2.0, 3.0])
pointer = (ctypes.c_float * len(values)).from_buffer(values)
library.scale(pointer, len(values), 2.0)
print(values)
```

`ctypes` 简单直接，但类型、长度和生命周期主要靠调用者保证。签名写错可能导致崩溃，而不只是普通 Python 异常。

### 6.3 `pybind11`：自然地绑定 C++ 接口

`pybind11` 能把 C++ 函数、类和异常映射成 Python 对象：

```cpp
#include <pybind11/pybind11.h>

int add(int lhs, int rhs) {
    return lhs + rhs;
}

PYBIND11_MODULE(vector_ops, module) {
    module.doc() = "minimal vector ops extension";
    module.def("add", &add, pybind11::arg("lhs"), pybind11::arg("rhs"));
}
```

绑定大型数组时要特别核对：

- dtype 是否匹配；
- shape 和 stride 是否符合算法假设；
- 数据是否连续；
- 内存位于 CPU 还是 GPU；
- 谁拥有底层存储，返回后是否仍有效；
- 计算期间是否应该释放 GIL；
- 原生异常如何转换为 Python 异常。

### 6.4 零拷贝不是“没有条件”

所谓零拷贝通常只是共享同一块底层内存。它要求双方对以下契约达成一致：

```text
地址 + 元素类型 + 形状 + 步长 + 设备 + 生命周期 + 可变性
```

任何一项不匹配都可能导致隐式复制、错误结果或越界访问。比如转置后的 NumPy 数组可能不是 C 连续布局；只拿首地址并按连续数组读取会得到错误数据。

> **核心原则**：把数据布局和所有权当作接口的一部分，而不是实现细节。

---

## 7. Linux 开发基本功

### 7.1 文件、目录与文本流

```bash
pwd                         # 当前目录
ls -lah                     # 包括隐藏文件的详细列表
mkdir -p runs/exp01/logs     # 递归创建目录
cp -a configs runs/exp01/   # 保留属性地复制
mv old_name new_name
rm -i file                  # 交互式删除；危险命令先确认路径

find runs -type f -name '*.log'
rg 'CUDA error|out of memory' runs
```

Shell 的组合能力来自标准输入、标准输出和标准错误：

```bash
python train.py >train.log 2>train.err       # 分别重定向
python train.py >train.all.log 2>&1           # 合并输出
rg 'loss=' train.log | tail -n 20
```

管道把左侧标准输出连接到右侧标准输入。每一段都应完成一个清晰任务，复杂且需要维护的流程应写成脚本并纳入版本控制。

### 7.2 引号、变量与退出码

```bash
name='experiment 01'
printf '%s\n' "$name"   # 双引号保留为一个参数
printf '%s\n' '$name'   # 单引号不展开变量
```

未加引号的变量可能发生单词分割和通配符展开。Shell 脚本中几乎总应写 `"$variable"`。

每个命令都有退出码：`0` 表示成功，非零表示失败。

```bash
python check_data.py
status=$?
printf 'exit code: %s\n' "$status"
```

健壮的 Bash 脚本常以此开头：

```bash
#!/usr/bin/env bash
set -euo pipefail
```

- `-e`：未处理的失败使脚本退出；
- `-u`：使用未定义变量时报错；
- `pipefail`：管道中任一命令失败，整个管道失败。

这些选项并不能替代清晰的错误处理，尤其要理解条件语句和管道中的例外语义。

### 7.3 权限模型

```bash
ls -l script.sh
# -rwxr-x---  owner group ... script.sh

chmod u+x script.sh          # 给所有者增加执行权限
chmod 640 config.yaml        # owner: rw, group: r, other: none
```

`r/w/x` 对文件分别表示读、写、执行；对目录则表示列出名称、修改目录项、穿越目录。权限问题先用 `ls -l`、`id` 和父目录权限定位，不要习惯性 `chmod 777`。

### 7.4 进程、作业与信号

```bash
ps -ef | rg train.py
pgrep -af train.py
top                         # 或 htop

python train.py &           # 后台运行
jobs
fg %1                       # 拉回前台
```

发送信号：

```bash
kill -TERM <pid>            # 请求进程优雅退出
kill -INT <pid>             # 类似 Ctrl-C
kill -KILL <pid>            # 内核立即终止，无法清理；最后手段
```

优先使用 `SIGTERM`，让程序有机会保存 checkpoint、释放共享资源和刷新日志。`SIGKILL` 无法被捕获或清理。

长期远程任务可使用 `tmux` 或集群调度系统，而不是仅依赖 SSH 终端：

```bash
tmux new -s train
# Ctrl-b d：退出但保持会话
tmux attach -t train
```

### 7.5 查看 CPU、内存、磁盘和网络

| 问题 | 常用命令 |
|------|----------|
| CPU/负载 | `top`、`uptime`、`mpstat` |
| 进程内存 | `ps`、`top`、`pmap -x PID` |
| 系统内存 | `free -h`、`vmstat` |
| 磁盘容量 | `df -h` |
| 目录占用 | `du -sh PATH`、`du -h --max-depth=1` |
| 文件 I/O | `iostat`、`iotop` |
| 监听端口 | `ss -lntp` |
| 连接测试 | `curl -v URL`、`nc -vz HOST PORT` |

“内存不够”至少可能指主机 RAM、GPU 显存、共享内存 `/dev/shm`、进程地址空间限制或磁盘满。先确认资源种类再行动。

### 7.6 环境变量与动态库路径

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

env | sort
which nvcc
type -a python
```

环境变量由父进程传给子进程。修改 `~/.bashrc` 不会神奇地改变已经运行的终端、IDE 或服务；需要重新加载配置或重启相应进程。

使用 `env -i` 可以在近乎空白的环境中复现问题，但要显式补充必要变量：

```bash
env -i HOME="$HOME" PATH=/usr/bin:/bin bash --noprofile --norc
```

### 7.7 SSH 与文件传输

生成密钥并配置登录：

```bash
ssh-keygen -t ed25519 -C "your-name"
ssh-copy-id user@server
ssh user@server
```

`~/.ssh/config` 可保存别名：

```text
Host gpu-dev
    HostName 192.0.2.10
    User lxy
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

之后直接 `ssh gpu-dev`。同步大目录优先使用支持增量传输的 `rsync`：

```bash
rsync -av --progress ./project/ gpu-dev:~/project/
```

同步前仔细检查源路径末尾的 `/`，它会影响复制“目录本身”还是“目录内容”。敏感数据和私钥不要放进仓库或随意传到共享机器。

### 7.8 调试原生程序

编译时加入调试信息：

```bash
g++ -g -O0 main.cpp -o app
gdb ./app
```

GDB 基本操作：

```text
break main       # 设置断点
run              # 启动
next / step      # 单步，是否进入函数
print variable   # 查看变量
bt               # 查看调用栈
continue         # 继续运行
```

优化版崩溃也应保留符号信息，例如 `-O2 -g`，这样 core dump 才有可读调用栈。调试器回答“在哪里崩”，Sanitizer 更擅长回答“哪次非法内存操作造成了它”。

系统调用层的问题可以用 `strace`（Linux）观察：

```bash
strace -f -o trace.log ./app
rg 'ENOENT|EACCES' trace.log
```

这对定位“程序到底在找哪个配置文件/动态库”非常有效。

---

## 8. 环境隔离与 GPU 软件栈

### 8.1 `venv`、conda 与容器解决不同层次的问题

| 工具 | 隔离范围 | 适合场景 |
|------|----------|----------|
| `venv` | Python 包与解释器环境 | 纯 Python 项目、依赖简单 |
| conda | Python + 一部分原生库与工具链 | 科学计算、复杂二进制依赖 |
| Docker/容器 | 用户态文件系统、依赖与进程环境 | 部署、CI、跨机器复现 |

`venv` 示例：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

始终用 `python -m pip` 可以减少“运行的是 A 环境的 Python，却调用 B 环境的 pip”这类混淆。

conda 示例：

```bash
conda create -n ai-infra python=3.12
conda activate ai-infra
conda env export --from-history > environment.yml
```

锁定直接依赖版本有助于复现，但完整环境还包括操作系统、驱动、编译器、硬件架构和启动参数。不要把一个 `requirements.txt` 当成全部环境描述。

### 8.2 容器不是虚拟机

容器与宿主机共享内核，只隔离用户态文件系统、进程、网络等资源。GPU 容器通常仍使用**宿主机 NVIDIA 驱动**，再通过 NVIDIA Container Toolkit 把设备和驱动库暴露给容器。

```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD:/workspace" \
  -w /workspace \
  nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
  nvidia-smi
```

镜像 tag 应固定到足够具体的版本。卷挂载会让容器内进程直接修改宿主目录，运行陌生命令前要理解挂载和用户权限。

### 8.3 GPU 软件栈：四个容易混淆的部分

```text
应用 / PyTorch / 自定义扩展
        │
CUDA Runtime 与数学库（cudart、cuBLAS、cuDNN 等）
        │
用户态驱动库（libcuda）
        │
内核态 NVIDIA Driver
        │
GPU
```

- **NVIDIA Driver**：操作系统与 GPU 通信的基础；`nvidia-smi` 主要反映驱动状态；
- **CUDA Toolkit**：开发工具集合，包含 `nvcc`、头文件、调试/分析工具和库；
- **CUDA Runtime/数学库**：应用运行时加载的用户态组件；
- **cuDNN**：面向深度学习算子的独立加速库，不等于整个 CUDA。

`nvidia-smi` 显示的“CUDA Version”通常表示驱动可支持的最高 CUDA 兼容级别，并不证明本机安装了同版本 `nvcc`。判断编译工具链要看 `nvcc --version`，判断框架实际使用的运行时要看框架自身信息。

### 8.4 诊断 GPU 环境的固定顺序

```bash
# 1. 驱动是否看到设备
nvidia-smi

# 2. 编译工具链来自哪里
which nvcc
nvcc --version

# 3. Python 与框架来自哪里
which python
python -c 'import sys; print(sys.executable)'

# 4. 框架看到什么
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("built with CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY
```

常见现象的含义：

| 现象 | 优先检查 |
|------|----------|
| `nvidia-smi` 失败 | 驱动、设备挂载、宿主机状态 |
| `nvidia-smi` 成功但找不到 `nvcc` | Toolkit 未装或 `PATH` 错；运行预编译框架未必需要 `nvcc` |
| 框架导入成功但 CUDA 不可用 | 安装了 CPU 构建、驱动兼容、容器未传 GPU、设备被屏蔽 |
| 自定义扩展编译失败 | 编译器、Toolkit、头文件、ABI、目标 GPU 架构 |
| 运行时报缺 `.so` | 动态库搜索路径、wheel/conda 依赖、架构不匹配 |

一次只改变一个变量，并保留原始报错。大多数环境问题都能通过这条链路定位，无需“全删重装”。

---

## 9. Git 协作与问题定位方法

### 9.1 一条干净的贡献流程

```bash
git clone https://github.com/owner/project.git
cd project
git switch -c docs/improve-foundations

# 修改并验证
git status --short
git diff --check
git diff

git add path/to/changed-file
git commit -m "docs: expand programming foundations"
```

提交应只包含一个清晰主题。不要把格式化整个仓库、编辑器配置和目标改动混在一起。

### 9.2 读懂工作区状态

Git 需要区分三层：

```text
HEAD 中的提交  ← git commit ← 暂存区 ← git add ← 工作区
```

```bash
git diff                 # 工作区 vs 暂存区
git diff --staged        # 暂存区 vs HEAD
git status --short
git log --oneline --decorate -10
```

在提交前同时看 `git diff` 与 `git diff --staged`，可避免漏提交或把无关文件带入。

### 9.3 合并、变基与冲突

- `merge` 保留两条分支历史并创建合并提交；
- `rebase` 把本分支提交重新播放到新基线，历史线性，但会改写提交 ID。

个人尚未共享的功能分支可以 rebase；公共分支不要随意改写历史。冲突解决不是机械删除标记，而是理解双方意图、保留正确组合并重新运行测试。

### 9.4 一个可复用的定位框架

遇到问题时按以下顺序记录：

1. **期望行为**：应该发生什么；
2. **实际行为**：完整报错、退出码、错误输出；
3. **最小复现**：最短命令、最小输入；
4. **环境**：代码提交、依赖版本、系统、硬件；
5. **边界定位**：数据、Python、C++、CUDA、驱动、网络中的哪一层；
6. **单变量实验**：每次只改变一项并记录结果。

“偶尔 OOM”不是可执行的问题描述；“提交 abc123、batch size 8、序列长度 4096，在第 23 step 分配 1.2 GiB 时 CUDA OOM，batch size 4 不复现”才是。

---

## 10. 综合实战：给 Python 添加一个 C++ 算子

下面用 `pybind11` 实现逐元素缩放，完整走过源码、构建、导入和验证。这个例子只为了展示跨语言链路；真正的数值计算应优先使用成熟向量化库。

### 10.1 项目结构

```text
fast_scale/
├── CMakeLists.txt
├── scale.cpp
└── test_scale.py
```

### 10.2 C++ 实现与绑定

```cpp
// scale.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

std::vector<float> scale(const std::vector<float>& input, float factor) {
    std::vector<float> output;
    output.reserve(input.size());
    for (float value : input) {
        output.push_back(value * factor);
    }
    return output;
}

PYBIND11_MODULE(fast_scale, module) {
    module.def("scale", &scale,
               pybind11::arg("input"),
               pybind11::arg("factor"));
}
```

这里使用 `const std::vector<float>&` 避免函数内部再复制输入，但 Python 列表转换成 `std::vector` 时仍会产生一次边界转换；返回值也会转换为 Python 列表。这不是零拷贝。

### 10.3 CMake 构建

```cmake
cmake_minimum_required(VERSION 3.20)
project(fast_scale LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(fast_scale scale.cpp)
target_compile_features(fast_scale PRIVATE cxx_std_17)
```

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pybind11

cmake -S . -B build \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### 10.4 验证正确性

```python
# test_scale.py
import sys

sys.path.insert(0, "build")
import fast_scale

actual = fast_scale.scale([1.0, -2.0, 3.5], 2.0)
expected = [2.0, -4.0, 7.0]
assert actual == expected
print(actual)
```

```bash
python test_scale.py
```

### 10.5 从教学例子到高性能接口还差什么

这个例子没有比 Python 列表推导式更有优势，因为边界上发生了逐元素转换。真正的高性能扩展还要考虑：

1. 接收 NumPy/PyTorch 连续缓冲区而不是 Python 列表；
2. 校验 dtype、shape、stride 与 device；
3. 避免不必要复制，并明确所有权；
4. 大计算期间释放 GIL；
5. 为 CPU 向量化或 CUDA 提供批量实现；
6. 对比可信参考实现，测试边界输入；
7. 分开测量边界转换和核心计算。

这正是后续 PyTorch 扩展、CUDA Kernel 和算子优化章节要继续解决的问题。

---

## 总结

本章的核心不是三套孤立语法，而是一套跨层思维：

- Python 中要看清对象引用、数据复制、并发模型和解释器/原生边界；
- C++ 中要看清内存布局、生命周期、所有权、编译链接和 ABI；
- Linux 中要看清进程、资源、路径、权限、动态库和环境继承；
- 性能问题先测量再优化，环境问题沿软件栈逐层验证；
- 一个好接口不仅描述“传什么值”，还应描述类型、形状、布局、设备、所有权与生命周期。

后续学习 CUDA、PyTorch、分布式训练和推理框架时，你会不断遇到这些概念，只是它们换成了 GPU 显存、Stream、Tensor、通信进程和模型服务的形式。

## 自我检验清单

- [ ] 能解释 `is` 与 `==`、浅拷贝与深拷贝的区别
- [ ] 能说明默认可变参数为什么会跨调用共享
- [ ] 能写一个保留函数元信息的装饰器和一个上下文管理器
- [ ] 能根据 I/O 密集、CPU 密集选择线程、进程或异步 I/O
- [ ] 知道 GPU 异步执行为什么会让普通 CPU 计时失真
- [ ] 能区分 C++ 指针、引用、值和 `const` 引用的接口语义
- [ ] 能说明 RAII、`unique_ptr`、`shared_ptr` 各自解决什么问题
- [ ] 能解释预处理、编译、汇编、链接和动态装载阶段
- [ ] 能用 CMake 构建一个最小 C++ 库
- [ ] 能解释 shape、stride、dtype、device 和 lifetime 为什么属于跨语言接口
- [ ] 能在 Linux 上查进程、资源占用、端口和动态库依赖
- [ ] 能区分 NVIDIA Driver、CUDA Toolkit、CUDA Runtime 与 cuDNN
- [ ] 能创建小而清晰的 Git 分支和提交，并在提交前检查 diff
- [ ] 能用“最小复现 + 环境快照 + 单变量实验”定位问题

## 练习题

1. 写一段代码复现浅拷贝共享内层列表的问题，并分别用数据重构和深拷贝修复。
2. 实现带参数的 `@retry(max_attempts=3)` 装饰器，只捕获调用方指定的异常类型。
3. 分别用线程池和进程池执行 I/O 等待任务、纯 Python CPU 任务，记录并解释耗时差异。
4. 用 `cProfile` 分析一个脚本，找出累计耗时最高的三个函数；优化其中一个并重新测量。
5. 写一个持有文件句柄的 C++ RAII 类，禁用复制并支持移动。
6. 故意制造一次堆越界，分别观察普通运行和 AddressSanitizer 的报错差异。
7. 把一个 C++ 库拆成头文件、源文件和可执行程序，用 target-based CMake 构建。
8. 用 `ldd`/`otool -L` 与 `nm -C` 分析一个 Python 原生扩展的依赖和导出符号。
9. 在干净虚拟环境中安装 PyTorch，记录 `sys.executable`、框架版本、构建 CUDA 版本和设备可见性。
10. 改造综合实战：接收 NumPy 数组，检查 dtype 与连续性，并比较复制版和共享缓冲区版的耗时。

## 参考资料

- [Python 官方教程](https://docs.python.org/3/tutorial/)
- [Python Data Model](https://docs.python.org/3/reference/datamodel.html)
- [Python `concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html)
- [Python `asyncio`](https://docs.python.org/3/library/asyncio.html)
- [Python Profilers](https://docs.python.org/3/library/profile.html)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- [CMake 官方教程](https://cmake.org/cmake/help/latest/guide/tutorial/)
- [pybind11 官方文档](https://pybind11.readthedocs.io/)
- [GNU GDB 文档](https://sourceware.org/gdb/documentation/)
- [Git 官方文档](https://git-scm.com/doc)
- [Docker 官方文档](https://docs.docker.com/)
- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
