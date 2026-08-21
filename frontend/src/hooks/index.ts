import useSWR from 'swr';
import { fetchKnowledgeBases } from '@/api/kb';
import { fetchModels, fetchCurrentModelConfig } from '@/api/models';

export function useKnowledgeBases() {
  return useSWR('/knowledge-bases', () => fetchKnowledgeBases(), {
    revalidateOnFocus: false,
    dedupingInterval: 5000,
  });
}

export function useModels(modelType?: string) {
  return useSWR(modelType ? `/models?type=${modelType}` : '/models', () => fetchModels(modelType), {
    revalidateOnFocus: false,
  });
}

export function useCurrentModelConfig() {
  return useSWR('/models/current-config', () => fetchCurrentModelConfig(), {
    revalidateOnFocus: false,
    dedupingInterval: 10000,
  });
}
