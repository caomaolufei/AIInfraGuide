---
title: "快手 AI Infra 实习 一面 (3)"
description: "快手 AI Infra 实习一面面试真题，涵盖推理优化、训练优化、算子优化等方向"
pubDate: 2026-04-16
company: "快手"
tier: "T1"
interviewType: "实习"
round: "一面"
order: 2009
tags: ["推理优化", "训练优化", "算子优化"]
---

Kstar 大模型训练/推理岗  50分钟，面完直接躺地上睡觉zzz

上来直接对着简历里的实习经历和项目问。

1. 有哪些KV Cache Compression的形式？（这边我想先从自己的实习项目开始，再说别的，被对方直接打断“我没有问你的东西，我问有哪些方法”  态度感觉很不友好 ）  开始吟唱八股。

2. MHA，MQA，GQA的概念，问怎样广播KV。之后问Multi-head Latent Attention与GQA的数据对应关系（给定hidden_status，Rope，MLA和GQA个数问MLA对应几个GQA），只知道MLA是低秩矩阵乘，但是具体怎么算没自己看过……没答上来。

3. 问了20分钟项目里的KV Cache Sparse计算的细节和vLLM Triton的实现，自我感觉答得还可以。不过其中有一个，问我KV Cache Sparse计算为什么不用掩码，跟他说用掩码会导致不必要的GPU I/O和计算，不如直接传入稀疏矩阵，但对方一直觉得我说的有问题 = =||  

4. DeepSpeed Zero123分别做了什么工作，吟唱完Zero1后被直接打断，让我算如果用Adam优化器，N个参数量的规模下Zero1如何给P个GPU分配数据。磕磕绊绊答出来，但是被说N个参数量还要考虑不同数据类型之间占用的内存不同FP32FP16balabala

5. SmoothQuant原理，为什么要Smooth，参数如何设定（八股启动）  怎样判断一个模型是否适合SmoothQuant，如果用每层激活值分布判断，是看input channel还是output channel（答output，但是说完之后对方不置可否 = =||）

6. AWQ和GPTQ原理，有何区别。

7. 项目里为什么选用不同的量化方法，GPTQ和SmoothQuant对应什么场景。

8. 蒸馏模型怎么做的，用了哪些技术（因为我用的模型是训练组给的蒸馏模型，只知道蒸馏的概念，细节不清楚）

9. 分布式gpu通信原语  all together   all2all  （展开说了分别各自对应什么场景，结果被打断说“我只需要知道你告诉我这是通信原语就行”  觉得我说太多了…急着下班吗）

反问环节有点幽默……我“请问您这边主要是做上游的微调或者modeling还是偏模型工程的推理加速？”  对方沉默一会儿回答“我们是算法”  把我尬住半天，，，

面完1分钟看官网秒挂
