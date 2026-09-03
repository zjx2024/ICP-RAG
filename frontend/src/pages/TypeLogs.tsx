import { useParams } from 'react-router-dom';
import ParseRecordsPanel from '@/components/ParseRecordsPanel';

/**
 * 兼容旧页面组件；产品入口已收进文档页的“解析记录”抽屉。
 * 保留该组件供历史引用复用，数据请求和展示只维护在 ParseRecordsPanel 中。
 */
export default function TypeLogs() {
  const { docType = '' } = useParams<{ docType: string }>();

  return (
    <div className="glass-fade-in type-page type-logs-page">
      <ParseRecordsPanel docType={docType} />
    </div>
  );
}
