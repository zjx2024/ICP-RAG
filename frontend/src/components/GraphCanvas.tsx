import { useEffect, useRef, useState } from 'react';
import G6 from '@antv/g6';
import type { GraphVertex, GraphEdge } from '@/types';

// 实体类型映射（黑白主题：统一灰阶，靠明度梯度 + 图标区分类型）
export const ENTITY_TYPE_MAP: Record<string, { color: string; icon: string; name: string }> = {
  FAULT: { color: '#1a1a1a', icon: '⚡', name: '故障' },
  CAUSE: { color: '#3d3d3d', icon: '🔍', name: '原因' },
  ACTION: { color: '#595959', icon: '✅', name: '措施' },
  CHECK: { color: '#737373', icon: '🔧', name: '检查' },
  CATEGORY: { color: '#8f8f8f', icon: '📁', name: '分类' },
  SYMPTOM: { color: '#a6a6a6', icon: '⚠️', name: '症状' },
  EQUIPMENT: { color: '#4d4d4d', icon: '⚙️', name: '设备' },
  PART: { color: '#333333', icon: '🔩', name: '部件' },
  default: { color: '#bfbfbf', icon: '📌', name: '其他' },
};

const LAYOUTS: Record<string, any> = {
  force: {
    type: 'force',
    preventOverlap: true,
    nodeSpacing: 40,
    linkDistance: 300,
    nodeSize: 60,
    repulsion: 700,
    gravity: 0.02,
    edgeStrength: 0.6,
    damping: 0.3,
    interval: 16,
    maxIteration: Infinity,
  },
  circular: { type: 'circular', radius: 280, divisions: 5, spacing: 80 },
  dagre: { type: 'dagre', rankdir: 'TB', nodesep: 90, ranksep: 160 },
  radial: { type: 'radial', radius: 300, linkDistance: 220, preventOverlap: true, nodeSpacing: 40, nodeSize: 60 },
};

export interface GraphCanvasProps {
  vertices: GraphVertex[];
  edges: GraphEdge[];
  layout: string;
  onNodeClick?: (node: any) => void;
  onEdgeClick?: (edge: any) => void;
  onCanvasClick?: () => void;
  highlightStates?: { highlightIds: Set<string>; dimIds: Set<string>; color?: string };
}

export default function GraphCanvas({
  vertices,
  edges,
  layout,
  onNodeClick,
  onEdgeClick,
  onCanvasClick,
  highlightStates,
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);
  const [resizeTick, setResizeTick] = useState(0);

  // 渲染图谱
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    if (graphRef.current) {
      graphRef.current.destroy();
      graphRef.current = null;
    }
    if (vertices.length === 0) return;

    const nodes = vertices.map((v) => {
      const name = (v.properties?.name as string) || v.label || v.id;
      const rawType = ((v.properties?.entity_type as string) || '').toUpperCase();
      const tm = ENTITY_TYPE_MAP[rawType] || ENTITY_TYPE_MAP.default;
      return {
        id: v.id,
        label: `${tm.icon} ${name}`,
        type: v.label || v.type,
        entityType: rawType,
        properties: v.properties,
        style: {
          fill: tm.color,
          stroke: '#ffffff',
          lineWidth: 1.5,
          shadowColor: tm.color,
          shadowBlur: 6,
          shadowOffsetX: 0,
          shadowOffsetY: 0,
        },
        labelCfg: { position: 'bottom', style: { fill: '#262626', fontSize: 11 } },
      };
    });

    const edgeList = edges.map((e) => ({
      source: e.source || e.outV,
      target: e.target || e.inV,
      label:
        (e.properties?.relation_type as string) ||
        (e.properties?.description as string) ||
        e.label ||
        '',
      properties: e.properties,
      type: 'cubic-bezier',
      style: { stroke: '#a0a0a0', lineWidth: 2, endArrow: G6.Arrow.triangle(8, 10, 0), lineAppendWidth: 10 },
      labelCfg: { style: { fill: '#6b7a99', fontSize: 10 } },
    }));

    const graph = new G6.Graph({
      container,
      width: container.offsetWidth || 800,
      height: container.offsetHeight || 600,
      fitView: true,
      fitViewPadding: 20,
      animate: true,
      modes: {
        default: ['drag-canvas', 'zoom-canvas', 'drag-node', 'click-select', 'brush-select'],
      },
      defaultNode: { size: 50, type: 'circle' },
      defaultEdge: { type: 'polyline' },
      layout: LAYOUTS[layout] || LAYOUTS.force,
      nodeStateStyles: {
        highlight: { fill: '#d9d9d9', stroke: '#595959', lineWidth: 3, shadowBlur: 12 },
        dim: { opacity: 0.2, shadowBlur: 0 },
        selected: { stroke: '#1a1a1a', lineWidth: 3 },
      },
      edgeStateStyles: {
        highlight: { stroke: '#1a1a1a', lineWidth: 3 },
        dim: { opacity: 0.08 },
      },
    });

    graph.data({ nodes, edges: edgeList } as any);
    graph.render();
    // 布局收敛后再自适应缩放，确保整张图谱完整落于视口内
    graph.on('afterlayout', () => {
      try {
        graph.fitView([40, 40]);
      } catch {
        /* ignore */
      }
    });

    if (onNodeClick) {
      graph.on('node:click', (evt: any) => onNodeClick(evt.item.getModel()));
    }
    if (onEdgeClick) {
      graph.on('edge:click', (evt: any) => onEdgeClick(evt.item.getModel()));
    }
    if (onCanvasClick) {
      graph.on('canvas:click', () => onCanvasClick());
    }

    graphRef.current = graph;
    return () => {
      if (graphRef.current) {
        graphRef.current.destroy();
        graphRef.current = null;
      }
    };
  }, [vertices, edges, layout, resizeTick]);

  // 应用高亮/dim 状态
  useEffect(() => {
    const graph = graphRef.current;
    if (!graph || !highlightStates) return;
    const { highlightIds, dimIds, color } = highlightStates;
    const data = graph.get('data') as any;
    const nodes = data.nodes || [];
    nodes.forEach((node: any) => {
      const model = graph.findById(node.id);
      if (!model) return;
      if (highlightIds.has(node.id)) {
        graph.setItemState(node.id, 'highlight', true);
        graph.setItemState(node.id, 'dim', false);
        if (color) {
          model.update({ style: { stroke: color, lineWidth: 4 } });
        }
      } else if (dimIds.has(node.id)) {
        graph.setItemState(node.id, 'highlight', false);
        graph.setItemState(node.id, 'dim', true);
      } else {
        graph.setItemState(node.id, 'highlight', false);
        graph.setItemState(node.id, 'dim', false);
        model.update({ style: { stroke: '#ffffff', lineWidth: 1.5 } });
      }
    });
    graph.paint();
  }, [highlightStates]);

  // 容器尺寸变化时重设
  useEffect(() => {
    const handler = () => setResizeTick((t) => t + 1);
    window.addEventListener('resize', handler);
    return () => window.removeEventListener('resize', handler);
  }, []);

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%', minHeight: 0, background: 'transparent', position: 'relative' }}
    />
  );
}
