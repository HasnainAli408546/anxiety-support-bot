"""
Therapeutic Knowledge Base for RAG Implementation
Curates evidence-based therapeutic content for 7-scenario anxiety support system
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import logging
from dotenv import load_dotenv
import os

# Load the .env file just once (at module import time)
load_dotenv()
CHROMADB_PATH = os.getenv("CHROMADB_PATH")

class TherapeuticKnowledgeBase:
    """
    RAG knowledge base with multiple scenario collections (techniques, education, reassurance, resources, scripts)
    """
    COLLECTIONS = ['techniques', 'education', 'reassurance', 'resources']

    def __init__(self, db_path: str = None, model_name: str = "all-MiniLM-L6-v2"):
        # Use env path if not provided (allows safe override for testing)
        db_path = db_path or CHROMADB_PATH or "therapeutic_kb"
        print(f"[DEBUG] Using Chromadb path: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        self.encoder = SentenceTransformer(model_name)
        self.logger = self._setup_logger()
        self.collections = {name: self._get_or_create_collection(name) for name in self.COLLECTIONS}
        self.logger.info(f"Knowledge base initialized with collections: {list(self.collections.keys())}")

    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(self.__class__.__name__)

    def _get_or_create_collection(self, name: str):
        try:
            return self.client.get_collection(name)
        except Exception:
            self.logger.info(f"Creating new collection: {name}")
            return self.client.create_collection(name)

    def add_content(self, content_type: str, content: str, metadata: Dict[str, Any]) -> str:
        if content_type not in self.collections:
            raise ValueError(f"Unknown content type: {content_type}")
        embedding = self.encoder.encode(content).tolist()
        collection = self.collections[content_type]
        content_id = f"{content_type}_{collection.count() + 1}_{datetime.now().strftime('%Y%m%d')}"
        enriched_metadata = {**metadata, "id": content_id, "content_type": content_type, "added_date": datetime.now().isoformat()}
        collection.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[enriched_metadata],
            ids=[content_id]
        )
        self.logger.info(f"Added content to {content_type}: {content_id}")
        return content_id

    def search_content(
        self,
        query: str,
        content_type: str,
        filters: Optional[Dict[str, Any]] = None,
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        if content_type not in self.collections:
            self.logger.warning(f"Unknown content type: {content_type}")
            return []
        query_embedding = self.encoder.encode(query).tolist()
        collection = self.collections[content_type]
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                where=filters if filters else {},
                n_results=n_results,
                include=["documents", "metadatas", "distances", "ids"]
            )
            if not results or not results.get("documents") or not results["documents"][0]:
                return []
            return [
                {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "id": results["ids"][0][i],
                    "relevance_score": 1 - results["distances"][0][i]
                }
                for i in range(len(results["documents"][0]))
            ]
        except Exception as e:
            self.logger.error(f"Error searching content: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict[str, int]:
        stats = {}
        for name, collection in self.collections.items():
            try:
                stats[name] = collection.count()
            except Exception as e:
                stats[name] = 0
        return stats

    def backup_knowledge_base(self, backup_path: str) -> bool:
        try:
            backup_data = {"created": datetime.now().isoformat(), "collections": {}}
            for name, collection in self.collections.items():
                all_data = collection.get()
                backup_data["collections"][name] = {
                    "documents": all_data.get("documents", []),
                    "metadatas": all_data.get("metadatas", []),
                    "ids": all_data.get("ids", [])
                }
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Knowledge base backed up to {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False
