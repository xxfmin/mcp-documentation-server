# MCP Documentation Server

Python MCP (Model Context Protocol) server for document indexing and semantic search.

## Structure

```
server/
├── src/              # Source code
│   ├── core/         # Core functionality (indexing, document management)
│   ├── lib/          # Shared utilities, config, models, errors
│   ├── retrieval/    # Search engine
│   ├── storage/      # Data storage (collections, documents, vectors)
│   └── server.py     # MCP server entry point
├── data/             # Data directory (auto-created)
│   ├── collections/  # Collection metadata
│   ├── documents.db  # Document metadata database
│   └── lancedb/      # Vector database
└── uploads/          # Upload directory for batch processing
```

## Running the Server

### Option 1: Using fastmcp (Recommended)

From the `server/` directory:

```bash
cd server
fastmcp dev src/server.py
```

Or from the `server/src/` directory:

```bash
cd server/src
fastmcp dev server.py
```

### Option 2: Using Python directly

From the `server/src/` directory:

```bash
cd server/src
python server.py
```

Or using Python module syntax from `server/`:

```bash
cd server
python -m src.server
```

### Option 3: Using the convenience script

From the `server/` directory:

```bash
cd server
python run.py
```

## Configuration

Configuration is managed via environment variables (see `src/lib/config.py`):

- `MCP_BASE_DIR`: Base directory (defaults to `server/` directory)
- `MCP_EMBEDDING_MODEL`: Embedding model (default: "Qwen/Qwen3-Embedding-0.6B")
- `HF_TOKEN`: HuggingFace token for model access
- `MCP_INDEXING_ENABLED`: Enable document indexing (default: true)
- `MCP_CACHE_ENABLED`: Enable embedding cache (default: true)
- `MCP_PARALLEL_ENABLED`: Enable parallel processing (default: true)

## Data Paths

All data is stored relative to `BASE_DIR` (defaults to `server/` directory):

- Collections: `server/data/collections/`
- Documents DB: `server/data/documents.db`
- Vector DB: `server/data/lancedb/`
- Uploads: `server/uploads/`

## Testing Imports

Run the test script to verify all imports work:

```bash
cd server
python test_imports.py
```
