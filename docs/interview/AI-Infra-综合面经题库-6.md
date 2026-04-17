---
title: "AI Infra 综合面经题库 (6)"
description: "AI Infra 面试真题，涵盖推理优化、训练优化、算子优化等方向"
pubDate: 2026-04-16
company: "综合"
tier: "综合"
interviewType: "未知"
order: 8
tags: ["推理优化", "训练优化", "算子优化"]
---

模型部署/推理优化/高性能计算方向社招面经总结

背景：lz在自驾公司做bev模型在orin上的部署和优化，三年工作经验。最近面了几家公司，主要是自驾公司的模型部署和工程化岗位，也有一些是推理框架和大模型推理优化相关的岗位。这两周大概搞了二十多场面试，也该总结一下经验，顺便攒一波rp了:)

lz既然是社招，那肯定是项目问的比较多，这块没啥好讲的，每个人的项目都不一样，只要保证简历上写的东西是自己做的，面试的时候能讲清楚就可以了。

CUDA和C++八股

lz的cuda基本都是自学的，自我感觉良好，结果面试的时候被各路面试官吊打，还是自己太菜了cuda的问题有下面这些

cuda graph作用原理，kernel launch流程
如何确定blocksize和gridsize
什么是default stream，它存在什么问题
shared memor的bank conflict及解决方法
threadfence的作用
如何debug cuda kernel
unified memory和zero-copy memory
cuda sort如何实现
sin函数在哪个硬件计算，这个硬件还能算什么
Volat架构特性，ITS
3090上单个block能用的shmem最大有多少
PTX与SASS的区别
GPU性能xx TFLOPS是怎么计算的

C++的八股问的也挺多，不过翻来覆去就下面几个问题

C++虚函数实现机制，单继承、多继承、虚继承的内存布局
四种cast
三种智能指针
函数模板声明与定义能否分离
CRTP静态多态
vector扩容，resize和reserve
单例模式
手撕题目

做推理优化和高性能计算肯定是要懂点cuda，所以大部分的题目都是用cuda实现，一些不太好用cuda实现的如NMS就用c++写了。当然也遇到过一些力扣题目，基本是hot100的范畴，这里不再赘述。

cuda实现：reduction，softmax，matrix transpose，avg pooling，算两堆bbox的iou，大部分情况下都是实现kernel即可，少数情况需要跟cpu对齐。

c++实现：NMS，conv2d，双线性插值，layernorm，单例模式

这里面让我印象比较深刻的是layernorm，用cuda写个layernorm不难，但面试官让我用vadd/vsub/vmul/vdiv等向量指令实现一个layernorm，我人都傻了。一是咱平时写cuda都是SIMT的编程模型，cpu优化是SIMD，这俩写起来有差别；二是没提供sqrt，得自己用牛顿法求，而且还没有比较运算符，浮点数的比较还有一些trick，最后肯定是寄了。

另外就是某大模型公司，要求实现softmax，需要跟cpu版本对齐。我写了个3pass的softmax，可惜面试过程中结果没有对齐，面完下来5分钟就解了bug，也算比较可惜吧。还有个小插曲就是面试官让我把代码通过腾讯会议发给他，但是我发错文件了，目录下有test.cc和test.cu，test.cc里面是另一个家面试的手撕，test.cu里面才是softmax。手滑了，发给面试官的是test.cc第二天我重新发了调通的test.cu，希望他能看到吧
