export const GUIDE_CATEGORIES = {
  'learning-path': { label: 'AIInfra学习路线', icon: '🗺️', order: 1 },
  'prerequisites': { label: 'AIInfra前置基础', icon: '📚', order: 2 },
  'cuda-optimization': { label: 'CUDA编程与算子优化', icon: '⚡', order: 3 },
  'distributed-training': { label: '分布式训练', icon: '🌐', order: 4 },
  'inference-optimization': { label: '推理优化', icon: '🚀', order: 5 },
  'performance-analysis': { label: '性能分析', icon: '📊', order: 6 },
} as const;

export type CategorySlug = keyof typeof GUIDE_CATEGORIES;
