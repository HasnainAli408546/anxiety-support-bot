"""
Memory-Enhanced RAG Integration
Combines user memory with RAG system for personalized content retrieval
"""

import os
import sys
from typing import Dict, List

# Allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from memory.user_memory import UserMemorySystem
    from rag.content_retriever import ContentRetriever
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("Warning: Could not import memory or rag modules. Using mock implementations.")

    class UserMemorySystem:
        def get_personalization_context(self, user_id: str) -> Dict:
            return {'has_sufficient_data': False}

    class ContentRetriever:
        def retrieve_for_scenario(self, **kwargs) -> List[Dict]:
            return [{'content': kwargs.get('query',''), 'metadata': {}, 'relevance_score': 0.5}]

class MemoryEnhancedRAG:
    """
    RAG system enhanced with user memory for personalized content retrieval
    """
    def __init__(self):
        if DEPENDENCIES_AVAILABLE:
            self.memory_system = UserMemorySystem()
            self.content_retriever = ContentRetriever()
            self.active = True
        else:
            self.memory_system = UserMemorySystem()
            self.content_retriever = ContentRetriever()
            self.active = False

    def retrieve_personalized_content(self,
                                      user_id: str,
                                      query: str,
                                      scenario: str,
                                      n_results: int = 5) -> List[Dict]:
        """
        Retrieve content personalized based on user memory and preferences

        Returns a list of dicts with keys: content, metadata, relevance_score, personalized_relevance
        """
        # If personalization not active, fallback to plain retrieval
        if not self.active:
            return self.content_retriever.retrieve_for_scenario(
                scenario=scenario, query=query, n_results=n_results
            )

        # 1. Get user context
        context = self.memory_system.get_personalization_context(user_id)

        # 2. Enhance query with top effective techniques
        enhanced_query = self._enhance_query_with_memory(query, context)

        # 3. Retrieve embeddings from RAG
        results = self.content_retriever.retrieve_for_scenario(
            scenario=scenario, query=enhanced_query, n_results=n_results
        )

        # 4. Rank by user patterns
        ranked = self._rank_content_by_user_patterns(results, context)
        return ranked

    def _enhance_query_with_memory(self, query: str, context: Dict) -> str:
        """
        Enhance the RAG query using user memory context—primarily effective techniques.
        """
        prefs = context.get('preferences', {})
        eff = prefs.get('effective_techniques', {})
        # pick top 2
        tops = sorted(eff.items(), key=lambda x: x[1], reverse=True)[:2]
        techs = [t for t,s in tops if s>0.6]
        if techs:
            return f"{query} {' '.join(techs)}"
        return query

    def _rank_content_by_user_patterns(self,
                                       content: List[Dict],
                                       context: Dict) -> List[Dict]:
        """
        Apply a simple boost to relevance scores of items containing effective techniques
        and preferred scenarios.
        """
        prefs = context.get('preferences', {})
        eff = prefs.get('effective_techniques', {})
        pref_scenarios = prefs.get('preferred_scenarios', [])

        for item in content:
            boost = 0.0
            text = item.get('content','').lower()
            meta = item.get('metadata',{})
            # boost by technique matches
            for t,score in eff.items():
                if t in text:
                    boost += score * 0.1
            # boost by scenario preference
            scen = meta.get('scenario')
            if scen in pref_scenarios:
                boost += 0.05
            item['personalized_relevance'] = item.get('relevance_score',0.5) + boost
        # sort descending
        return sorted(content, key=lambda x: x.get('personalized_relevance',0), reverse=True)
