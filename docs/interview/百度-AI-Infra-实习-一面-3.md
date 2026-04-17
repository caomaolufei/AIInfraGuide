---
title: "百度 AI Infra 实习 一面 (3)"
description: "百度 AI Infra 实习一面面试真题，涵盖推理优化、算子优化等方向"
pubDate: 2026-04-16
company: "百度"
tier: "T0"
interviewType: "实习"
round: "一面"
order: 1029
tags: ["推理优化", "算子优化"]
---

百度提前批（直接开始二战） 高性能计算工程师一面面经

没想到吧兄弟们，直接开始二战了。捞了我就面呗～这回面的挺爽的。

点名表扬语音部门，面试至少感觉respect。

八股/经历
自我介绍：懂得都懂，开源+实习
讲了讲在字节的实习工作：大模型训练模拟器
根据这个他问了我TP PP DP都是什么，具体流程
如何根据TP PP的通信量进行取舍
问了量化相关，什么是per tensor，per channel，group wise
不同的量化方法之间的区别，为什么group wise能更加降低量化误差
不同的量化之间开销的区别，如何降低开销
你了解量化中的异常值吗，如何消除异常值（提到了下面三种方法
大模型量化方法介绍 gptq awq smoothquant都是什么
你了解kv cache量化吗 请讲讲kv cache量化
你知道flash attn吗，flash attn为什么会加速？
flash attn 1 2 之间有什么区别
flash attn 每个Bc块的切分思路是什么，1 loop flash attn是怎么做的
你知道paged attention吗，思路是什么？
大模型prefill 阶段和 decoding阶段的区别是什么，为什么会有这种区别
prefill阶段的flash attn在decoding阶段会有什么问题，decoding阶段的attn方法（flash decoding
讲讲flash decoding的思路。
讲到了具体组件：RmsNorm是如何实现的
如果现在你要优化一个cuda kernel 你的优化思路是什么？
现在有一个conv2d 它的输入CWH是[64, 64, 128] 卷积核大小是3x3 它的输出大小是[128, 64, 128]。问它的参数量是多大，计算flops是多大（flops算晕了没算出来
如果你现在要用cpu做算子优化，你知道该怎么做吗（把我知道的avx512都说了，笑死
c++ 八股 智能指针
做题
cuda layernorm

--------------------------------------------------

后续，秒过了。好好好好好好好好好，你比paddle大大滴好
