import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timezone
import logging

from lib import Collection, Config

logger = logging.getLogger(__name__)

"""Collection management for organizing documents"""
class CollectionManager:
    def __init__(self, data_dir: str = "data/collections"):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Config.BASE_DIR / "data" / "colletions"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CollectionManager initialized with data_dir: {self.data_dir}")

    def create_collection(
            self,
            collection_id: str,
            description: Optional[str] = None,
            embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
            chunk_size: int = 512,
            chunk_overlap: int = 50
    ) -> Collection:
        """
        Create a new collection.

        Args:
            collection_id: Unique identifier for the collection
            description: Fuman-readable description
            embedding_model: embedding model to use for this collection
            chunk_size: Maximum chink size in tokens
            chunk_overlap: Overlap between chunks in tokens

        Returns:
            Created Collection object

        Raises:
            Value Error: If collection already exists
        """
        if self.collection_exists(collection_id):
            raise ValueError(f"Collection '{collection_id}' already exists")
        
        collection = Collection(
            collection_id=collection_id,
            description=description,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            created_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )

        self.save_collection(collection)
        logger.info(f"Created collection: {collection_id}")
        return collection
    
    def get_collection(self, collection_id: str) -> Optional[Collection]:
        """
        Get collection by ID.

        Args:
            collection_id: Collection identifier

        Returns:
            Collection object if found, None otherwise
        """
        path = self.data_dir / f"{collection_id}.json"
        if not path.exists():
            logger.debug(f"Collection not found: {collection_id}")
            return None

        try:
            with open(path) as f:
                data = json.load(f)

            # Convert datetime strings back to datetime objects
            if "created_at" in data:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            if "last_updated" in data:
                data["last_updated"] = datetime.fromisoformat(data["last_updated"])
            
            return Collection(**data)
        except(json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to load collection {collection_id}: {e}")
            return None
    
    def list_collections(self) -> List[Collection]:
        """
        List all collections.

        Returns:
            List of Collection objects, sorted by last_updated (newest first)
        """
        collections = []
        for path in self.data_dir.glob("*.json"):
            try:
                collection = self.get_collection(path.stem)
                if collection:
                    collections.append(collection)
            except Exception as e:
                logger.warning(f"Failed to load collection from {path}: {e}")
                continue

        return sorted(collections, key=lambda c: c.last_updated, reverse=True)

    def update_collection(self, collection: Collection) -> None:
        """
        Update collection metadata.

        Args:
            collection: Collection object with updated data
        """
        if not self.collection_exists(collection.collection_id):
            raise ValueError(f"Collection '{collection.collection_id}' does not exist")

        collection.last_updated = datetime.now(timezone.utc)
        self.save_collection(collection)
        logger.info(f"Updated collection: {collection.collection_id}")

    def delete_collection(self, collection_id: str) -> bool:
        """
        Delete a collection.

        Args:
            collection_id: Collection identifier

        Returns:
            True if deleted, False if collection didn't exist
        """
        path = self.data_dir / f"{collection_id}.json"
        if path.exists():
            path.unlink()
            logger.info(f"Deleted collection: {collection_id}")
            return True
        return False

    def collection_exists(self, collection_id: str) -> bool:
        """
        Check if collection exists.

        Args:
            collection_id: Collection identifier

        Returns:
            True if collection exists, False otherwise
        """
        return (self.data_dir / f"{collection_id}.json").exists()

    def increment_document_count(self, collection_id: str, amount: int = 1) -> None:
        """
        Increment document count for a collection.

        Args:
            collection_id: Collection identifier
            amount: Amount to increment by (default: 1)
        """
        collection = self.get_collection(collection_id)
        if collection:
            collection.document_count += amount
            self.update_collection(collection)

    def increment_chunk_count(self, collection_id: str, amount: int = 1) -> None:
        """
        Increment chunk count for a collection.

        Args:
            collection_id: Collection identifier
            amount: Amount to increment by (default: 1)
        """
        collection = self.get_collection(collection_id)
        if collection:
            collection.chunk_count += amount
            self.update_collection(collection)

    def save_collection(self, collection: Collection) -> None:
        """
        Save collection to disk.

        Args:
            collection: Collection object to save
        """
        path = self.data_dir / f"{collection.collection_id}.json"
        try:
            with open(path, "w") as f:
                json.dump(
                    collection.model_dump(mode="json"),
                    f,
                    indent=2,
                    default=str,
                )
        except Exception as e:
            logger.error(f"Failed to save collection {collection.collection_id}: {e}")
            raise