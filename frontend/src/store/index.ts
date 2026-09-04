import { create } from 'zustand';
import type { KnowledgeBase, DocumentStatus, LLMModel, CurrentModelConfig } from '@/types';

// 全局共享状态（原 index.html 的全局变量）
interface GlobalState {
  knowledgeBases: KnowledgeBase[];
  setKnowledgeBases: (kbs: KnowledgeBase[]) => void;
  documents: DocumentStatus[];
  setDocuments: (docs: DocumentStatus[]) => void;
  models: LLMModel[];
  setModels: (models: LLMModel[]) => void;
  modelConfigStatus: CurrentModelConfig['config_status'] | null;
  setModelConfigStatus: (s: CurrentModelConfig['config_status']) => void;
}

export const useGlobalStore = create<GlobalState>((set) => ({
  knowledgeBases: [],
  setKnowledgeBases: (kbs) => set({ knowledgeBases: kbs }),
  documents: [],
  setDocuments: (docs) => set({ documents: docs }),
  models: [],
  setModels: (models) => set({ models }),
  modelConfigStatus: null,
  setModelConfigStatus: (s) => set({ modelConfigStatus: s }),
}));
