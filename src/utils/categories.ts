export const GUIDE_CATEGORIES = {
  'gpu-hardware': { label: 'GPU 与硬件', icon: '🖥️', order: 1 },
  'training-frameworks': { label: '训练框架', icon: '🏋️', order: 2 },
  'inference-serving': { label: '推理与服务', icon: '⚡', order: 3 },
  'distributed-systems': { label: '分布式系统', icon: '🌐', order: 4 },
  'mlops': { label: 'ML Ops', icon: '🔧', order: 5 },
  'networking': { label: '网络通信', icon: '📡', order: 6 },
  'storage-data': { label: '存储与数据', icon: '💾', order: 7 },
  'scheduling': { label: '调度与编排', icon: '📋', order: 8 },
} as const;

export type CategorySlug = keyof typeof GUIDE_CATEGORIES;
