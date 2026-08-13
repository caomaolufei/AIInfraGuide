import type { CategorySlug } from './categories';

export interface ModuleMeta {
  /** 模块中文标题 */
  title: string;
  /** 简短描述 */
  description: string;
  /** 主题色（与 GuideSidebar 的 gradientMap 一致） */
  color: string;
  /** 学习建议 */
  tips: string[];
  /** 前置要求（简短描述） */
  prerequisites?: string;
}

export const MODULE_METADATA: Partial<Record<CategorySlug, ModuleMeta>> = {
  prerequisites: {
    title: '模块一：前置知识',
    description:
      '涵盖 GPU 架构、编程语言基础、数学基础、Transformer 架构、PyTorch 框架和集合通信等核心前置知识，为后续深入 AI Infra 打好坚实基础。',
    color: '#06B6D4',
    tips: [
      '建议按章节顺序学习，每章内容相对独立但有递进关系',
      '如果已有相关背景知识，可以跳过对应章节直接进入 CUDA 编程模块',
      'Transformer 架构和 PyTorch 框架是后续所有模块的基础，建议重点掌握',
    ],
  },
  'cuda-optimization': {
    title: '模块二：CUDA 编程与算子优化',
    description:
      '从 CUDA 编程入门到经典算子实现（Reduce、GEMM、Softmax、Attention），再到 AI 编译器和性能分析工具链，系统掌握 GPU 编程与算子优化技术。',
    color: '#F59E0B',
    tips: [
      '建议先完成前置知识模块中的 GPU 硬件概论',
      '每个算子实现章节都有配套代码，建议动手实践',
      '性能分析工具链是贯穿整个模块的重要工具，建议尽早熟悉',
    ],
    prerequisites: '前置知识模块（尤其是第5章 GPU 硬件概论）',
  },
  'distributed-training': {
    title: '模块三：分布式训练',
    description:
      '从分布式训练总论出发，深入数据并行、ZeRO 系列、张量并行、流水线并行、3D 并行策略，最终通过训练框架实战串联全部知识。',
    color: '#10B981',
    tips: [
      '分布式训练总论提供全局视角，建议首先阅读',
      'ZeRO 系列是理解现代分布式训练的关键，建议重点学习',
      '训练框架实战章节会将前面的理论知识串联起来',
    ],
    prerequisites: '前置知识模块（尤其是第6章 集合通信基础）',
  },
  'inference-optimization': {
    title: '模块四：推理优化',
    description:
      '覆盖 LLM 推理基础、vLLM 与 SGLang、量化、Speculative Decoding、分布式推理、P/D 解耦、生产服务特性、性能分析、部署运维与端到端实战。',
    color: '#8B5CF6',
    tips: [
      'LLM 推理基础章节帮助理解推理场景的独特挑战',
      '量化和 Speculative Decoding 是当前最热门的推理优化方向',
      '建议用同一 workload 分别部署 vLLM 与 SGLang，以统一指标验证选型',
    ],
    prerequisites: '前置知识模块 + CUDA 编程基础',
  },
};

/** category slug → URL 前缀映射 */
export const CATEGORY_URL_PREFIX: Partial<Record<CategorySlug, string>> = {
  prerequisites: '/prerequisites',
  'cuda-optimization': '/cuda',
  'distributed-training': '/distributed',
  'inference-optimization': '/inference',
};
