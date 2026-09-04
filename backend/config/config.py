import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenSearch is the single search projection: BM25/full-text and the
    # 1024-dimensional BGE-M3 vector live in the same versioned index.
    OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
    OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
    OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME", "")
    OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "")
    OPENSEARCH_SCHEME = os.getenv("OPENSEARCH_SCHEME", "http")
    OPENSEARCH_VERIFY_CERTS = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"
    OPENSEARCH_INDEX_ALIAS = os.getenv("OPENSEARCH_INDEX_ALIAS", "knowledge_graph_default")
    OPENSEARCH_INDEX_VERSION = os.getenv("OPENSEARCH_INDEX_VERSION", "v1")
    OPENSEARCH_REQUEST_TIMEOUT = int(os.getenv("OPENSEARCH_REQUEST_TIMEOUT", 60))
    # Compose enables this so production never continues on the legacy in-memory
    # PostgreSQL/HugeGraph fallbacks.  Unit tests may leave it disabled.
    STRICT_DEPENDENCIES = os.getenv("STRICT_DEPENDENCIES", "false").lower() == "true"
    
    # 原始文档、工程图和解析产物统一通过 ArtifactStore 访问。业务代码只保存
    # storage_key，不依赖物理目录；当前实现使用本地持久化文件系统。
    ARTIFACT_BACKEND = os.getenv("ARTIFACT_BACKEND", "local")
    ARTIFACT_STORAGE_DIR = os.getenv("ARTIFACT_STORAGE_DIR", os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "volumes", "artifacts"
    ))
    ARTIFACT_TRASH_ENABLED = os.getenv("ARTIFACT_TRASH_ENABLED", "true").lower() == "true"
    
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    # 不写死默认口令：本地开发由 start_server.sh 从 docker compose 继承，
    # 生产（STRICT_DEPENDENCIES=true）必须通过环境变量显式提供。
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "knowledge_graph")
    # 强口令开关：置 true 后，STRICT 模式拒绝空口令或等于旧默认值 "password"
    # 的口令，强制部署方显式注入真实凭据。默认 false 保持与既有数据卷兼容。
    ICP_REQUIRE_STRONG_POSTGRES_PASSWORD = (
        os.getenv("ICP_REQUIRE_STRONG_POSTGRES_PASSWORD", "false").lower() == "true"
    )
    
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
    
    # 仅公开 ArtifactStore 的 images 命名空间，绝不暴露原始文档目录。
    IMAGE_STORAGE_DIR = os.getenv(
        "IMAGE_STORAGE_DIR", os.path.join(ARTIFACT_STORAGE_DIR, "images")
    )
    IMAGE_URL_PREFIX = os.getenv("IMAGE_URL_PREFIX", "/images")

    # B1 全链路 tracing 的 JSON-lines 落盘目录。
    TRACE_DIR = os.getenv("ICP_TRACE_DIR", os.path.join(ARTIFACT_STORAGE_DIR, "traces"))

config = Config()
