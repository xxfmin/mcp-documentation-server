import os
from pathlib import Path
from typing import Optional

# Get default data directory
def get_default_data_dir() -> Path:
    base_dir = os.getenv("MCP_BASE_DIR")
    if base_dir:
        return Path(base_dir).expanduser()
    # Default to server root directory
    # Go up 3 levels to get to server root: server/src/lib/ -> server/src/ -> server/
    project_root = Path(__file__).parent.parent.parent.resolve()
    return project_root

# Configuration from environment variables
class Config:
    # Base directory
    BASE_DIR = get_default_data_dir()

    # Embedding (key, model)
    HF_TOKEN = os.getenv("HF_TOKEN")
    EMBEDDING_MODEL = os.getenv("MCP_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

    # Feature flags
    INDEXING_ENABLED = os.getenv("MCP_INDEXING_ENABLED", "true").lower() == "true"
    CACHE_ENABLED = os.getenv("MCP_CACHE_ENABLED", "true").lower() == "true"
    CACHE_SIZE = int(os.getenv("MCP_CACHE_SIZE", "1000"))
    PARALLEL_ENABLED = os.getenv("MCP_PARALLEL_ENABLED", "true").lower() == "true"
    STREAMING_ENABLED = os.getenv("MCP_STREAMING_ENABLED", "true").lower() == "true"

    # Streaming configuration
    STREAM_STREAM_CHUNK_SIZE = int(os.getenv("MCP_STREAM_CHUNK_SIZE", "65536"))  # 64KB
    STREAM_FILE_SIZE_LIMIT = int(os.getenv("MCP_STREAM_FILE_SIZE_LIMIT", "10485760")) # 10MB

    # Parallel processing
    MAX_WORKERS = int(os.getenv("MCP_MAX_WORKERS", "4"))

    # Default chunk settings
    DEFAULT_CHUNK_SIZE = int(os.getenv("MCP_DEFAULT_CHUNK_SIZE", "512"))
    DEFAULT_CHUNK_OVERLAP = int(os.getenv("MCP_DEFAULT_CHUNK_OVERLAP", "50"))