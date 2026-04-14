# AIInfraGuide

> 深入理解 AI 基础设施的核心技术

**在线浏览：https://caomaolufei.github.io/AIInfraGuide/**

## 简介

AIInfraGuide 是一个开源的 AI 基础设施知识库与博客，系统梳理从 GPU 硬件到训练框架、从推理服务到分布式系统的完整技术栈，帮助工程师构建扎实的 AI Infra 知识体系。

## 知识库目录

| 分类 | 说明 |
|------|------|
| GPU 与硬件 | GPU 架构、CUDA 编程、显存管理、互联技术 |
| 训练框架 | PyTorch DDP/FSDP、Megatron-LM、DeepSpeed |
| 推理与服务 | vLLM、TensorRT-LLM、推理优化、量化 |
| 分布式系统 | 通信原语、并行策略、容错机制 |
| ML Ops | 实验管理、模型部署、监控告警 |
| 网络通信 | RDMA、InfiniBand、GPUDirect |
| 存储与数据 | 数据管道、检查点管理 |
| 调度与编排 | Kubernetes、GPU 调度策略 |

## 技术栈

- [Astro](https://astro.build/) - 静态站点生成
- [Tailwind CSS](https://tailwindcss.com/) - 样式框架
- [Pagefind](https://pagefind.app/) - 全文搜索
- [GitHub Pages](https://pages.github.com/) - 托管部署

## 本地开发

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 构建（含搜索索引生成）
npm run build

# 预览构建产物
npm run preview
```

## 添加内容

### 知识库文章

在 `src/content/guides/` 下创建 Markdown 文件：

```markdown
---
title: "文章标题"
description: "文章描述"
pubDate: 2024-01-15
category: "gpu-hardware"  # 见下方分类列表
order: 1
tags: ["GPU", "CUDA"]
---

正文内容...
```

可选分类：`gpu-hardware`、`training-frameworks`、`inference-serving`、`distributed-systems`、`mlops`、`networking`、`storage-data`、`scheduling`

### 博客文章

在 `src/content/posts/` 下创建 Markdown 文件：

```markdown
---
title: "文章标题"
description: "文章描述"
pubDate: 2024-01-15
tags: ["标签"]
---

正文内容...
```

## 部署

推送到 `main` 分支后，GitHub Actions 会自动构建并部署到 GitHub Pages。

> 首次部署需在仓库 Settings → Pages 中将 Source 设置为 **GitHub Actions**。

## 参与贡献

欢迎提交 Issue 和 Pull Request 来完善内容。

## License

MIT
