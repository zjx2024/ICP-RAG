from minio import Minio
from backend.config.config import config
import logging
import io

logger = logging.getLogger(__name__)


class MinioClient:
    def __init__(self):
        self.client = Minio(
            config.MINIO_ENDPOINT,
            access_key=config.MINIO_ACCESS_KEY,
            secret_key=config.MINIO_SECRET_KEY,
            secure=config.MINIO_SECURE
        )
    
    def upload_file(self, object_name, file_data):
        bucket_name = "documents"
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
            if hasattr(file_data, "read"):
                # 已是流对象：直接使用
                data = file_data
                data.seek(0, 2)
                size = data.tell()
                data.seek(0)
            else:
                # bytes：包装为 BytesIO（minio 对 bytes-like 的兼容更稳定）
                data = io.BytesIO(file_data)
                size = len(file_data)
            self.client.put_object(bucket_name, object_name, data, size)
        except Exception as e:
            logger.error(f"MinIO 上传失败 {bucket_name}/{object_name}: {e}")
    
    def get_file(self, object_name):
        bucket_name = "documents"
        try:
            response = self.client.get_object(bucket_name, object_name)
            return response.read()
        except Exception:
            with open(f"/tmp/{object_name.split('/')[-1]}", "rb") as f:
                return f.read()
    
    def delete_file(self, object_name):
        bucket_name = "documents"
        try:
            self.client.remove_object(bucket_name, object_name)
        except Exception:
            pass

minio_client = MinioClient()