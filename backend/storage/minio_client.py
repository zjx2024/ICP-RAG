"""Deprecated import shim for extensions written before ArtifactStore.

Production code imports :mod:`backend.storage.artifact_store` directly.  This
module deliberately has no MinIO SDK or network dependency.
"""

from backend.config.config import config
from backend.storage.artifact_store import LocalFSArtifactStore, artifact_store


class MinioClient(LocalFSArtifactStore):
    """No-network compatibility constructor for legacy extensions.

    The historical class was constructed without arguments.  Keep that shape
    while routing all operations to the configured LocalFS root.
    """

    def __init__(self):
        super().__init__(
            config.ARTIFACT_STORAGE_DIR,
            trash_enabled=config.ARTIFACT_TRASH_ENABLED,
        )


minio_client = artifact_store

__all__ = ["MinioClient", "minio_client"]
