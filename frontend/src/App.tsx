import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import KbLayout from './layouts/KbLayout';
import TypeLayout from './layouts/TypeLayout';
import TypeHome from './pages/TypeHome';
import { Spin } from 'antd';

// 懒加载页面（按文档 4.3 路由表）
const TypeDocs = lazy(() => import('./pages/TypeDocs'));
const TypeView = lazy(() => import('./pages/TypeView'));
const TypeLogs = lazy(() => import('./pages/TypeLogs'));
const TypeGraph = lazy(() => import('./pages/TypeGraph'));
const TypeVisualize = lazy(() => import('./pages/TypeVisualize'));
const DocDetail = lazy(() => import('./pages/DocDetail'));
const Upload = lazy(() => import('./pages/Upload'));
const Retrieval = lazy(() => import('./pages/Retrieval'));
const ViewDoc = lazy(() => import('./pages/ViewDoc'));
const GraphBuilder = lazy(() => import('./pages/GraphBuilder'));
const Visualize = lazy(() => import('./pages/Visualize'));
const Models = lazy(() => import('./pages/Models'));
const Logs = lazy(() => import('./pages/Logs'));
const Agent = lazy(() => import('./pages/Agent'));
const Gallery = lazy(() => import('./pages/Gallery'));

function Loading() {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
      <Spin size="large" />
    </div>
  );
}

export default function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          {/* 首页：7 类文档类型卡片 */}
          <Route index element={<TypeHome />} />
          {/* ICP 类型文档：类型侧边栏框架 + 子页面 */}
          <Route path="type/:docType" element={<TypeLayout />}>
            <Route index element={<Navigate to="upload" replace />} />
            <Route path="upload" element={<TypeDocs />} />
            <Route path="view" element={<TypeView />} />
            <Route path="logs" element={<TypeLogs />} />
            <Route path="graph" element={<TypeGraph />} />
            <Route path="visualize" element={<TypeVisualize />} />
            <Route path="doc/:docId" element={<DocDetail />} />
          </Route>
          {/* 全局页面 */}
          <Route path="models" element={<Models />} />
          <Route path="agent" element={<Agent />} />
          <Route path="gallery" element={<Gallery />} />
          {/* 知识库详情（二级菜单，kbId 由 URL 提供） */}
          <Route path="kb/:kbId" element={<KbLayout />}>
            <Route index element={<Navigate to="upload" replace />} />
            <Route path="upload" element={<Upload />} />
            <Route path="retrieval" element={<Retrieval />} />
            <Route path="view" element={<ViewDoc />} />
            <Route path="graph" element={<GraphBuilder />} />
            <Route path="visualize" element={<Visualize />} />
            <Route path="logs" element={<Logs />} />
            <Route path="gallery" element={<Gallery />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
