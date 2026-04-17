---
title: "AI Infra 综合面经题库 (6)"
description: "AI Infra 面试真题，涵盖推理优化、训练优化、算子优化等方向"
pubDate: 2026-04-17
company: "综合"
tier: "综合"
interviewType: "未知"
order: 8
tags: ["推理优化", "训练优化", "算子优化"]
---

## CUDA 基础知识

1. CUDA Graph 的作用与原理，kernel launch 的完整流程
2. 如何确定合适的 blockSize 与 gridSize？
3. 什么是 default stream？它存在哪些潜在问题？
4. Shared Memory 的 bank conflict 产生原因及解决方法
5. threadfence 的作用是什么？
6. 如何调试 CUDA kernel？
7. Unified Memory 与 Zero-Copy Memory 的区别
8. CUDA 中排序算法如何实现？
9. sin 函数在 GPU 的哪个硬件单元上执行？该单元还支持哪些运算？
10. Volta 架构的特性，ITS（Independent Thread Scheduling）的概念
11. 在 3090 上单个 block 可使用的 Shared Memory 最大容量是多少？
12. PTX 与 SASS 的区别是什么？
13. GPU 标称的 xx TFLOPS 性能指标是如何计算的？

## C++ 基础知识

14. C++ 虚函数的实现机制，单继承、多继承、虚继承下的内存布局
15. 四种类型转换（static_cast / dynamic_cast / const_cast / reinterpret_cast）的用法与区别
16. 三种智能指针（unique_ptr / shared_ptr / weak_ptr）的适用场景
17. 函数模板的声明与定义能否分离到不同文件？
18. CRTP（Curiously Recurring Template Pattern）实现静态多态的原理
19. vector 的扩容机制，resize 与 reserve 的区别
20. 单例模式的实现方式

## CUDA 编程题

1. Reduction 归约求和
2. Softmax 实现
3. 矩阵转置（Matrix Transpose）
4. 平均池化（Avg Pooling）
5. 计算两组 BBox 的 IoU

## C++ 编程题

1. NMS（非极大值抑制）
2. Conv2D 卷积实现
3. 双线性插值
4. LayerNorm 实现
5. 单例模式实现
