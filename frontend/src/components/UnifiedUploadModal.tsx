import { useEffect, useMemo, useState } from 'react';
import {
  Alert, Button, Modal, Select, Space, Tag, Typography, Upload, message,
} from 'antd';
import type { UploadFile } from 'antd';
import {
  CloudUploadOutlined, FileTextOutlined, InboxOutlined,
} from '@ant-design/icons';
import { uploadIcpDocuments } from '@/api/icp';
import { GlassButton } from '@/components/GlassUI';
import type { DocTypeInfo } from '@/types';

interface UnifiedUploadModalProps {
  open: boolean;
  types: DocTypeInfo[];
  initialDocType?: string;
  onClose: () => void;
  onUploaded: (docType: string, count: number) => void | Promise<void>;
}

export default function UnifiedUploadModal({
  open,
  types,
  initialDocType,
  onClose,
  onUploaded,
}: UnifiedUploadModalProps) {
  const [docType, setDocType] = useState('');
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    if (!open) return;
    const preferred = initialDocType && types.some((item) => item.doc_type === initialDocType)
      ? initialDocType
      : types[0]?.doc_type || '';
    setDocType(preferred);
    setFileList([]);
  }, [open, initialDocType, types]);

  const selectedType = useMemo(
    () => types.find((item) => item.doc_type === docType),
    [docType, types],
  );

  const close = () => {
    if (uploading) return;
    setFileList([]);
    onClose();
  };

  const confirmUpload = async () => {
    if (!docType) {
      message.warning('请先选择文档类型');
      return;
    }
    const files = fileList
      .map((item) => item.originFileObj)
      .filter(Boolean) as File[];
    if (files.length === 0) {
      message.warning('请选择要上传的文件');
      return;
    }

    setUploading(true);
    try {
      const result = await uploadIcpDocuments(docType, files);
      message.success(`已上传 ${result.length} 份文档，并归入“${selectedType?.name || docType}”`);
      setFileList([]);
      onClose();
      await onUploaded(docType, result.length);
    } catch (error) {
      message.error(`上传失败：${(error as Error).message}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <Modal
      title={
        <Space>
          <CloudUploadOutlined />
          <span>统一上传七类文档</span>
        </Space>
      }
      open={open}
      onCancel={close}
      width={620}
      maskClosable={!uploading}
      closable={!uploading}
      footer={[
        <Button key="cancel" onClick={close} disabled={uploading}>取消</Button>,
        <GlassButton
          key="upload"
          variant="primary"
          icon={<CloudUploadOutlined />}
          loading={uploading}
          disabled={!docType || fileList.length === 0}
          onClick={confirmUpload}
        >
          {uploading ? '正在上传…' : `上传${fileList.length ? `（${fileList.length}）` : ''}`}
        </GlassButton>,
      ]}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div>
          <Typography.Text strong>1. 选择文档类型</Typography.Text>
          <Select
            value={docType || undefined}
            placeholder="请选择 HMC、FTA、RBD 等文档类型"
            style={{ width: '100%', marginTop: 8 }}
            options={types.map((item) => ({
              value: item.doc_type,
              label: `${item.name}（${item.doc_type}）`,
            }))}
            onChange={(value) => {
              setDocType(value);
              setFileList([]);
            }}
          />
        </div>

        {selectedType ? (
          <Alert
            type="info"
            showIcon
            message={selectedType.name}
            description={
              <Space direction="vertical" size={4}>
                <span>{selectedType.description || '该类型暂无说明'}</span>
                <Space wrap size={[4, 4]}>
                  <Tag>{selectedType.data_form === 'table' ? '表格类' : selectedType.data_form === 'image' ? '图类' : '图文/文本类'}</Tag>
                  {selectedType.input_formats ? <Tag color="blue">支持 {selectedType.input_formats}</Tag> : null}
                </Space>
              </Space>
            }
          />
        ) : null}

        <div>
          <Typography.Text strong>2. 选择文件</Typography.Text>
          <Upload.Dragger
            multiple
            accept={selectedType?.input_formats || undefined}
            beforeUpload={() => false}
            fileList={fileList}
            onChange={({ fileList: next }) => setFileList(next)}
            onRemove={(file) => {
              setFileList((current) => current.filter((item) => item.uid !== file.uid));
              return true;
            }}
            style={{ marginTop: 8 }}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined style={{ fontSize: 44, color: 'rgba(0,0,0,.42)' }} />
            </p>
            <p className="ant-upload-text">点击或拖拽文件到此处</p>
            <p className="ant-upload-hint">
              文件会根据上方选择的类型写入对应知识域，并自动显示在相应卡片和文档列表中。
            </p>
          </Upload.Dragger>
        </div>

        <div style={{ color: '#8c8c8c', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
          <FileTextOutlined />
          上传只完成文档归类与登记；后续解析可在对应类型的“文档管理”页面启动。
        </div>
      </div>
    </Modal>
  );
}
