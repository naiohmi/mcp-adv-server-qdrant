import asyncio
import torch
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.base_transformer import BaseTransformerProvider

class TransformersProvider(EmbeddingProvider, BaseTransformerProvider):
    """
    Transformers implementation of the embedding provider for Qwen3-Embedding-0.6B.
    :param model_name: The name of the Transformers model to use.
    :param output_dim: Output embedding dimension (32-1024 for Qwen3-Embedding-0.6B).
    """

    def __init__(self, model_name: str, output_dim: int = 1024):
        BaseTransformerProvider.__init__(self, model_name)
        self.output_dim = min(max(output_dim, 32), 1024)  # Clamp between 32-1024

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._embed_documents_sync(documents)
        )
        return embeddings

    def _embed_documents_sync(self, documents: list[str]) -> list[list[float]]:
        """Synchronously embed documents using proper embedding extraction."""
        # Tokenize with proper truncation for 32k context length
        inputs = self.tokenizer(
            documents,
            padding=True,
            truncation=True,
            max_length=32768,  # 32k context length
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # For embedding models, use the last hidden state with proper pooling
            # Apply attention mask for proper pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Reduce dimension if needed
            if self.output_dim < embeddings.size(1):
                embeddings = embeddings[:, :self.output_dim]
            
        return embeddings.cpu().tolist()

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._embed_query_sync(query)
        )
        return embeddings

    def _embed_query_sync(self, query: str) -> list[float]:
        """Synchronously embed a single query."""
        # Tokenize with proper truncation
        inputs = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=32768,  # 32k context length
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Apply attention mask for proper pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Reduce dimension if needed
            if self.output_dim < embeddings.size(1):
                embeddings = embeddings[:, :self.output_dim]
            
        return embeddings.cpu().tolist()[0]

    def get_vector_name(self) -> str:
        """Return the name of the vector for the Qdrant collection."""
        model_name = self.model_name.split("/")[-1].lower()
        return f"transformers-{model_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        return self.output_dim
