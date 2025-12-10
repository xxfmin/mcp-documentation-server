import hashlib

# Sanitize filename by removing invalid characters
def sanitize_filename(filename: str) -> str:
    return filename.replace("/", "_").replace("\\", "_)").replace(":", "_")

# Generate stable document ID from source
def generate_document_id(source: str) -> str:
    return f"doc_{hashlib.md5(source.encode()).hexdigest()[:12]}"

# Generate stable chunk ID
def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    combined = f"{document_id}_{chunk_index}"
    return f"chunk_{hashlib.md5(combined.encode()).hexdigest()[:12]}"

# Format file size in human readable format
def format_file_size(bytes: int) -> str:
    sizes = ['Bytes', 'KB', 'MB', 'GB']
    if bytes == 0:
        return '0 Bytes'
    
    import math
    i = int(math.floor(math.log(bytes) / math.log(1024)))
    return f"{round(bytes / math.pow(1024, i), 2)} {sizes[i]}"