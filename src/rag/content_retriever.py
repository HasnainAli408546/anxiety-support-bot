import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import os
from dotenv import load_dotenv


load_dotenv()  # Loads from .env file automatically
CHROMADB_PATH = os.getenv("CHROMADB_PATH")


class ContentRetriever:
    """
    Retrieves contextual therapeutic content for flows from ChromaDB.
    Supports scenario-based filtering, emotion-aware retrieval, and personalization.
    """

    def __init__(
        self,
        db_path: str = os.getenv("CHROMADB_PATH"),
        model_name: str = "all-MiniLM-L6-v2",
        collections: List[str] = None,
    ):
        self.client = chromadb.PersistentClient(path=db_path)
        self.encoder = SentenceTransformer(model_name)
        self.logger = self._setup_logger()

        self.collections = {}
        self.collection_names = collections or ["techniques", "education", "reassurance", "resources"]

        for name in self.collection_names:
            try:
                self.collections[name] = self.client.get_collection(name)
            except Exception:
                self.logger.warning(f"Collection '{name}' not found. Creating empty collection.")
                self.collections[name] = self.client.create_collection(name)

        self.logger.info(f"Initialized retriever with collections: {list(self.collections.keys())}")

    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(self.__class__.__name__)

    def retrieve(
        self,
        scenario: str,
        query: str,
        content_type: str,
        filters: Optional[Dict[str, Any]] = None,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Flexible retrieval for scenario/content type with embedding-based search and filter.
        Tries with scenario filter first; if nothing is found, retries without scenario.
        """
        if content_type not in self.collections:
            self.logger.warning(f"Collection '{content_type}' does not exist in the KB.")
            return []

        full_query = f"{scenario} {query}".strip()
        embedding = self.encoder.encode(full_query).tolist()

        where = dict(filters) if filters else {}
        if scenario:
            where["scenario"] = scenario

        try:
            def _query(with_where: bool) -> Optional[Dict[str, Any]]:
                query_args = {
                    "query_embeddings": [embedding],
                    "n_results": n_results,
                    "include": ["documents", "metadatas", "distances"],
                }
                if with_where and where:
                    query_args["where"] = where
                return self.collections[content_type].query(**query_args)

            # 1) Try with scenario + filters
            results = _query(with_where=True)

            # 2) If empty, retry without scenario filter
            if (
                not results
                or not results.get("documents")
                or not results["documents"][0]
            ):
                if "scenario" in where:
                    where_no_scenario = dict(where)
                    where_no_scenario.pop("scenario", None)
                    where = where_no_scenario
                    results = _query(with_where=bool(where))

            if (
                not results
                or not results.get("documents")
                or not results["documents"][0]
            ):
                return []

            return [
                {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "id": results["metadatas"][0][i].get("id"),
                    "relevance_score": 1 - results["distances"][0][i],
                }
                for i in range(len(results["documents"][0]))
            ]

        except Exception as e:
            self.logger.error(f"Error retrieving for {scenario} - {query}: {str(e)}")
            return []

    def get_best(
        self,
        scenario: str,
        intent: str,
        content_type: str,
        user_context: Dict[str, Any] = None,
        fallback: Optional[str] = None,
    ) -> Optional[str]:
        """
        Retrieve best content matching scenario, intent, and context.

        Returns:
            - str: content string when a match is found
            - fallback: if provided and no match found
            - None: if no match and no fallback (caller decides what to do)
        """
        keywords = [intent]
        if user_context:
            for key, value in user_context.items():
                if isinstance(value, str):
                    keywords.append(value)
                elif isinstance(value, list):
                    keywords.extend(value)

        query = " ".join(keywords)
        results = self.retrieve(scenario, query, content_type, n_results=1)

        if results:
            return results[0]["content"]

        if fallback is not None:
            return fallback

        # Signal "no RAG match" to caller; do not leak a system string into user text
        return None

    def get_content_package(
        self,
        scenario: str,
        emotion_scores: Dict[str, float],
        user_context: Dict = None,
    ) -> Dict[str, str]:
        """
        Assemble a content package with best-fit technique, psychoeducation, and reassurance.
        """
        # Intensity estimation
        max_intensity = max(emotion_scores.values()) if emotion_scores else 0.5
        if max_intensity >= 0.8:
            intensity = "high"
        elif max_intensity >= 0.5:
            intensity = "medium"
        else:
            intensity = "low"

        technique = self.get_best(
            scenario,
            f"{intensity} coping technique",
            "techniques",
            user_context,
            fallback=(
                "Try a basic grounding exercise: pause, notice your breathing, "
                "and gently name 5 things you can see around you."
            ),
        )

        # Let this be RAG-first; if no result, return None and let flows decide
        education = self.get_best(
            scenario,
            "psychoeducation",
            "education",
            fallback=None,
        )

        reassurance = self.get_best(
            scenario,
            f"{intensity} reassurance",
            "reassurance",
            fallback=(
                "You’re not alone in feeling this way. Anxiety is a common human experience, "
                "and these feelings will pass with time and support."
            ),
        )

        return {
            "technique": technique or "",
            "education": education or "",
            "reassurance": reassurance or "",
            "intensity_used": intensity,
            "scenario": scenario,
            "personalization_applied": bool(user_context),
        }

    def search_by_keywords(
        self,
        keywords: List[str],
        content_type: str,
        scenario: Optional[str] = None,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Direct keyword-based semantic search for a given content type (collection).
        """
        query = " ".join(keywords)
        return self.retrieve(scenario or "", query, content_type, n_results=n_results)

    def get_stats(self) -> Dict[str, Any]:
        stats = {}
        for name, collection in self.collections.items():
            try:
                stats[name] = collection.count()
            except Exception:
                stats[name] = 0
        return stats

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Returns knowledge base stats and summary metadata for system analytics.
        """
        stats = self.get_stats()
        summary = {
            "total_items": sum(stats.values()),
            "collection_counts": stats,
            "scenarios": {},
            "content_types": {
                name: "active" if count > 0 else "empty" for name, count in stats.items()
            },
        }
        return summary

    # ----------- WRAPPER METHODS FOR CLINICAL FLOWS ------------

    def get_reassurance_content(self, scenario, confidence=0.5):
        """
        Retrieve reassurance content string for a scenario (confidence optionally controls type/intensity).
        """
        intensity = "low" if confidence < 0.5 else "medium" if confidence < 0.8 else "high"
        return self.get_best(scenario, f"{intensity} reassurance", "reassurance")

    def get_educational_content(self, scenario, topic="psychoeducation"):
        """
        Retrieve educational content string for a scenario and topic.
        """
        return self.get_best(scenario, topic, "education")

    def get_technique_for_scenario(self, scenario, intensity="medium", user_context=None):
        """
        Retrieve a technique for the scenario, optionally personalized by intensity and user context.
        """
        return self.get_best(
            scenario,
            f"{intensity} coping technique",
            "techniques",
            user_context,
        )

    # ------------- DIAGNOSTIC METHODS -------------
    def test_retrieval(self):
        """
        Simple diagnostic retrieval test—prints sample content for fixed scenarios.
        """
        scenarios = [
            ("panic", "breath technique", "techniques"),
            ("sleep", "insomnia help", "techniques"),
            ("isolation", "loneliness comfort", "reassurance"),
            ("uncertainty", "worry psychoeducation", "education"),
        ]
        for scenario, intent, ctype in scenarios:
            results = self.retrieve(scenario, intent, ctype, n_results=1)
            sample = results[0]["content"][:100] + "..." if results else "[Nothing found]"
            print(f"{scenario} | {ctype} | {intent}: {sample}")


if __name__ == "__main__":
    retriever = ContentRetriever()
    print(f"Collection stats: {retriever.get_stats()}")
    retriever.test_retrieval()
