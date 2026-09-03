import { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Spin } from 'antd';
import MainLayout from './layouts/MainLayout';
import TypeLayout from './layouts/TypeLayout';
import TypeHome from './pages/TypeHome';

const TypeDocs = lazy(() => import('./pages/TypeDocs'));
const TypeView = lazy(() => import('./pages/TypeView'));
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

          {/* 首页与类型页复用统一上传窗口；兼容上传路由不恢复独立页面。 */}
          <Route path="type/:docType" element={<TypeLayout />}>
            <Route index element={<Navigate to="documents" replace />} />
            <Route path="documents" element={<TypeDocs />} />
            <Route path="upload" element={<Navigate to="../documents" replace />} />
            <Route path="view" element={<TypeView />} />
            <Route path="logs" element={<Navigate to="../documents?records=1" replace />} />
            <Route path="graph" element={<Navigate to="../visualize" replace />} />
            <Route path="visualize" element={<TypeVisualize />} />
            <Route path="doc/:docId" element={<DocDetail />} />
          </Route>

          <Route path="diagnosis" element={<Diagnosis />} />
          <Route path="agent" element={<Navigate to="/diagnosis" replace />} />
          <Route path="models" element={<Models />} />

          <Route path="kb/*" element={<Navigate to="/" replace />} />
          <Route path="gallery" element={<Navigate to="/" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
