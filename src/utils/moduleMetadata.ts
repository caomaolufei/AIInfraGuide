import type { CategorySlug } from './categories';

export interface ModuleResource {
  /** 资源类型标签，如「练习平台」 */
  type: string;
  /** 资源名称 */
  title: string;
  /** 外部链接 */
  href: string;
  /** 一句话说明 */
  description: string;
}

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
  /** 延伸资源（站外链接） */
  resources?: ModuleResource[];
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
    resources: [
      {
        type: '练习平台',
        title: 'CUDA 手撕算子 · 面试练习',
        href: 'https://kkcocoa.github.io/cuda-operator-interview/',
        description:
          '覆盖访存、Reduce、Softmax、Attention、GEMM 等算子的分级手写练习，附提示、自检清单与浏览器内代码编辑器',
      },
      {
        type: '练习平台',
        title: 'LeetGPU',
        href: 'https://leetgpu.com/',
        description:
          'GPU 编程版的 LeetCode，按 easy/medium/hard 分级的算子题库，支持 CUDA、Triton、PyTorch 等多种写法，无需本地 GPU 即可提交运行，可查看 PTX/SASS 与性能排行',
      },
    ],
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
      '覆盖 LLM 推理基础、推理引擎核心技术、主流推理框架、量化、Speculative Decoding、PD 解耦架构、端侧推理，以及性能分析与端到端实战。',
    color: '#8B5CF6',
    tips: [
      'LLM 推理基础章节帮助理解推理场景的独特挑战',
      '量化和 Speculative Decoding 是当前最热门的推理优化方向',
      '建议结合主流推理框架（如 vLLM）进行实际部署练习',
      '端侧推理建议在真实设备上同时评估延迟、内存、功耗和精度',
    ],
    prerequisites: '前置知识模块 + CUDA 编程基础',
  },
  ascend: {
    title: '模块五：昇腾 AI 芯片与算子开发',
    description:
      '从昇腾硬件架构与达芬奇 AI Core 出发，深入 CANN 软件栈、AscendCL 应用开发、Ascend C 算子开发、HCCL 集合通信与分布式训练，再到 MindSpore 框架与 MindIE/vLLM-Ascend 推理，系统掌握国产 AI 算力全栈技术。',
    color: '#EF4444',
    tips: [
      '建议先掌握前置知识模块中的 GPU 硬件概论与集合通信基础，昇腾与 CUDA 体系高度同构，可对照学习',
      '第1~2章以「看懂+会用」为主，第3章 Ascend C 是算子开发核心，建议配合官方样例动手实践',
      '昇腾生态文档以昇腾社区（hiascend.com）与 CANN 官方文档为准，版本迭代快，注意核对版本匹配',
      '无真实硬件的同学可先学习原理并浏览官方样例，有条件时通过昇腾开发者云（如 HiDevLab）申请在线 NPU 资源',
    ],
    prerequisites: '前置知识模块（GPU 硬件概论、集合通信基础）+ 模块二 CUDA 编程基础（对照学习更佳）',
    resources: [
      {
        type: '官方社区',
        title: '昇腾社区',
        href: 'https://www.hiascend.com/',
        description: '昇腾官方开发者社区，提供文档、教程、硬件平台查询、CANN 下载与论坛支持',
      },
      {
        type: '官方文档',
        title: '昇腾文档中心',
        href: 'https://www.hiascend.com/document',
        description: 'CANN、AscendCL、Ascend C、MindIE 等全栈官方文档入口',
      },
      {
        type: '开源样例',
        title: 'Ascend Samples',
        href: 'https://github.com/Ascend/samples',
        description: '昇腾官方开源样例仓库，覆盖算子、推理、训练、C++/Python 全场景示例',
      },
      {
        type: '框架社区',
        title: '昇思 MindSpore',
        href: 'https://www.mindspore.cn/',
        description: '华为自研 AI 框架，与昇腾深度适配，提供大模型训练与推理套件',
      },
      {
        type: '推理生态',
        title: 'vLLM-Ascend',
        href: 'https://github.com/vllm-project/vllm-ascend',
        description: 'vLLM 官方社区维护的昇腾后端插件，让主流大模型推理无缝跑在昇腾 NPU 上',
      },
    ],
  },
};

/** category slug → URL 前缀映射 */
export const CATEGORY_URL_PREFIX: Partial<Record<CategorySlug, string>> = {
  prerequisites: '/prerequisites',
  'cuda-optimization': '/cuda',
  'distributed-training': '/distributed',
  'inference-optimization': '/inference',
  ascend: '/ascend',
};
