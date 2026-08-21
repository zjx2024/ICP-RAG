import { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Spin } from 'antd';
import MainLayout from './layouts/MainLayout';
import TypeLayout from './layouts/TypeLayout';
import TypeHome from './pages/TypeHome';

const TypeDocs = lazy(() => import('./pages/TypeDocs'));
const TypeView = lazy(() => import('./pages/TypeView'));
const TypeLogs = lazy(() => import('./pages/TypeLogs'));
const TypeGraph = lazy(() => import('./pages/TypeGraph'));
const TypeVisualize = lazy(() => import('./pages/TypeVisualize'));
const DocDetail = lazy(() => import('./pages/DocDetail'));
const Models = lazy(() => import('./pages/Models'));
const Diagnosis = lazy(() => import('./pages/Diagnosis'));

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
          <Route index element={<TypeHome />} />

          {/* 七类 ICP 文档的上传、解析、结构化查看与图谱确认。 */}
          <Route path="type/:docType" element={<TypeLayout />}>
            <Route index element={<Navigate to="upload" replace />} />
            <Route path="upload" element={<TypeDocs />} />
            <Route path="view" element={<TypeView />} />
            <Route path="logs" element={<TypeLogs />} />
            <Route path="graph" element={<TypeGraph />} />
            <Route path="visualize" element={<TypeVisualize />} />
            <Route path="doc/:docId" element={<DocDetail />} />
          </Route>

          <Route path="diagnosis" element={<Diagnosis />} />
          <Route path="agent" element={<Navigate to="/diagnosis" replace />} />
          <Route path="models" element={<Models />} />

          {/* 旧通用知识库与图片画廊页面已从 ICP 产品入口移除。 */}
          <Route path="kb/*" element={<Navigate to="/" replace />} />
          <Route path="gallery" element={<Navigate to="/" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
