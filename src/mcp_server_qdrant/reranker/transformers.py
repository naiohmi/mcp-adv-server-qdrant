import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from mcp_server_qdrant.reranker.base import RerankProvider
import torch
from typing import List, Tuple

class TransformersProvider(RerankProvider):
    """
    Transformers implementation of the reranker provider for Qwen3-Reranker-0.6B.
    :param model_name: The name of the Transformers reranker model to use.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure padding token is set for batch processing
        if self.tokenizer.pad_token is None:
            # Prefer eos_token, then unk_token, then add new pad token
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        # Ensure model config is aware of pad_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Set model to evaluation mode
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    async def reranking_result(self, query: str, documents: list[str]) -> List[Tuple[str, float]]:
        """
        Rerank a list of documents based on their relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            
        Returns:
            List of tuples (document, relevance_score) sorted by relevance score descending
        """
        loop = asyncio.get_event_loop()
        reranked_results = await loop.run_in_executor(
            None, lambda: self._reranking_result_sync(query, documents)
        )
        return reranked_results

    def _reranking_result_sync(self, query: str, documents: list[str]) -> List[Tuple[str, float]]:
        """Synchronously rerank documents using the reranker model."""
        if not documents:
            return []
        
        scores = []
        
        # Process documents in batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_scores = self._score_batch(query, batch_docs)
            scores.extend(batch_scores)
        
        # Combine documents with their scores and sort by score descending
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_score_pairs

    def _score_batch(self, query: str, documents: list[str]) -> list[float]:
        """Score a batch of documents against the query."""
        # Create query-document pairs for the reranker
        pairs = []
        for doc in documents:
            pairs.append([query, doc])
        
        # Tokenize the pairs
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=32768,  # 32k context length for Qwen3
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get the relevance scores
            # For reranking models, typically use the logits or a specific output
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                # If binary classification, take the positive class score
                if logits.shape[-1] == 2:
                    scores = torch.softmax(logits, dim=-1)[:, 1]  # Positive class probability
                else:
                    scores = torch.sigmoid(logits.squeeze(-1))  # Single output sigmoid
            else:
                # Fallback: use pooled output with sigmoid activation
                scores = torch.sigmoid(outputs.last_hidden_state.mean(dim=1).squeeze(-1))
        
        return scores.cpu().tolist()

    async def rerank_top_k(self, query: str, documents: list[str], top_k: int = None) -> List[Tuple[str, float]]:
        """
        Convenience method to rerank and return top-k results.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top results to return (None for all)
            
        Returns:
            List of top-k tuples (document, relevance_score)
        """
        reranked_results = await self.reranking_result(query, documents)
        
        if top_k is None:
            return reranked_results
        else:
            return reranked_results[:top_k]
