import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    ES_HOST = os.getenv("ES_HOST", "localhost")
    ES_PORT = int(os.getenv("ES_PORT", 9200))
    ES_USERNAME = os.getenv("ES_USERNAME", "")
    ES_PASSWORD = os.getenv("ES_PASSWORD", "")
    ES_SCHEME = os.getenv("ES_SCHEME", "http")
    
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password")
    MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
    
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "knowledge_graph")
    
    # 模型配置已迁移到数据库 llm_models 表
    # MODEL_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, QWEN_API_KEY, QWEN_API_URL, QWEN_MODEL, EMBEDDING_MODEL
    # 这些配置不再从 .env 读取，请在模型管理页面配置
    
    # 运行时参数（非模型身份），可保留作默认值
    MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", 0.7))
    MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", 262144))
    MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", 3000))
    MODEL_TOP_P = float(os.getenv("MODEL_TOP_P", 0.1))
    MODEL_ENABLE_THINKING = os.getenv("MODEL_ENABLE_THINKING", "false").lower() == "true"
    
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1024))
    
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    
    HUGEGRAPH_HOST = os.getenv("HUGEGRAPH_HOST", "localhost")
    HUGEGRAPH_PORT = int(os.getenv("HUGEGRAPH_PORT", 8081))
    HUGEGRAPH_GRAPH_NAME = os.getenv("HUGEGRAPH_GRAPH_NAME", "hugegraph")
    
    SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("SERVER_PORT", 9999))
    SERVER_IP = os.getenv("SERVER_IP", "").strip()
    
    IMAGE_STORAGE_DIR = os.getenv("IMAGE_STORAGE_DIR", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "images"
    ))
    IMAGE_URL_PREFIX = os.getenv("IMAGE_URL_PREFIX", "/images")

config = Config()