import { create } from 'zustand';
import type { LLMModel, CurrentModelConfig } from '@/types';

// 全局共享状态（原 index.html 的全局变量）
interface GlobalState {
  models: LLMModel[];
  setModels: (models: LLMModel[]) => void;
  modelConfigStatus: CurrentModelConfig['config_status'] | null;
  setModelConfigStatus: (s: CurrentModelConfig['config_status']) => void;
}

export const useGlobalStore = create<GlobalState>((set) => ({
  models: [],
  setModels: (models) => set({ models }),
  modelConfigStatus: null,
  setModelConfigStatus: (s) => set({ modelConfigStatus: s }),
}));
