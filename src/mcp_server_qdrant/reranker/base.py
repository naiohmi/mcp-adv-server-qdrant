from abc import ABC, abstractmethod
from typing import List, Tuple


class RerankProvider(ABC):
    """Abstract base class for reranker providers."""

    @abstractmethod
    async def reranking_result(self, query: str, documents: list[str]) -> List[Tuple[str, float]]:
        """
        Rerank a list of documents based on their relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            
        Returns:
            List of tuples (document, relevance_score) sorted by relevance score descending
        """
        pass