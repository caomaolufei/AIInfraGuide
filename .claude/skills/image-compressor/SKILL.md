---
name: image-compressor
description: >
  压缩图片文件体积，支持 PNG/JPG 等常见格式。
  触发场景：(1) 用户说"压缩图片"、"压缩图像"、"compress image"
  (2) 用户要求减小图片体积/文件大小
  (3) 用户要求优化图片用于博客/网页发布
  (4) 用户提供图片路径并要求压缩
---

# Image Compressor

使用 `scripts/compress.sh` 压缩图片。

## 压缩脚本用法

```bash
scripts/compress.sh <input> [--quality Q] [--resize W] [--format fmt] [--output path]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--quality` / `-q` | 压缩质量 (0-100，越小体积越小) | 40 |
| `--resize` / `-r` | 最大宽度/高度(px)，等比缩放 | 不缩放 |
| `--format` / `-f` | 目标格式: `png` / `jpg` | 保持原格式 |
| `--output` / `-o` | 输出路径 | 同目录，加 `_compressed` 后缀 |

## 工作流程

1. 用 Glob 定位图片文件，用 `ls -lh` 和 `sips -g pixelWidth -g pixelHeight` 获取原始大小和尺寸
2. 根据用户需求选择策略并运行 `scripts/compress.sh`
3. 用 Read 工具预览压缩结果，确认画质可接受
4. 向用户展示压缩前后对比表格
5. 确认后替换原文件或保留两份

## 压缩策略选择

- **PNG 保持 PNG**: 使用 pngquant（自动安装），`--quality 40` 通常可压缩 80%+
- **PNG/JPG 转 JPG**: 加 `--format jpg`，体积最小但有损且格式变更
- **缩小尺寸**: 加 `--resize 1024`，配合质量参数效果更好
- **极限压缩**: `--format jpg --quality 20 --resize 1024`

## 注意事项

- 压缩前始终确认原始文件是否已被 git 跟踪，未跟踪的文件替换后无法恢复
- 默认输出到新文件（不覆盖原文件），由用户决定是否替换
- PNG 格式适合图表/示意图（保留清晰文字），JPG 适合照片
