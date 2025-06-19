#!/usr/bin/env python3
"""
Test script for Qwen3 embedding and reranking models.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mcp_server_qdrant.embeddings.transformers import TransformersProvider as EmbeddingProvider
from mcp_server_qdrant.reranker.transformers import TransformersProvider as RerankerProvider


async def test_embedding_provider():
    """Test the embedding provider with Qwen3-Embedding-0.6B."""
    print("Testing Qwen3-Embedding-0.6B...")
    
    # Use a placeholder model name - replace with actual Qwen3 model name when available
    model_name = "Qwen/Qwen3-Embedding-0.6B"  # Replace with actual model name
    
    try:
        provider = EmbeddingProvider(model_name, output_dim=512)
        
        # Test documents
        documents = [
            "This is a test document about machine learning.",
            "Python is a popular programming language.",
            "Natural language processing is a subfield of AI."
        ]
        
        # Test query
        query = "What is machine learning?"
        
        print(f"Model name: {provider.model_name}")
        print(f"Vector name: {provider.get_vector_name()}")
        print(f"Vector size: {provider.get_vector_size()}")
        
        # Test embedding documents
        print("\nTesting document embedding...")
        doc_embeddings = await provider.embed_documents(documents)
        print(f"Embedded {len(doc_embeddings)} documents")
        print(f"Embedding dimension: {len(doc_embeddings[0])}")
        
        # Test embedding query
        print("\nTesting query embedding...")
        query_embedding = await provider.embed_query(query)
        print(f"Query embedding dimension: {len(query_embedding)}")
        
        print("✅ Embedding provider test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Embedding provider test failed: {e}")
        return False


async def test_reranker_provider():
    """Test the reranker provider with Qwen3-Reranker-0.6B."""
    print("\nTesting Qwen3-Reranker-0.6B...")
    
    # Use a placeholder model name - replace with actual Qwen3 model name when available
    model_name = "Qwen/Qwen3-Reranker-0.6B"  # Replace with actual model name
    
    try:
        provider = RerankerProvider(model_name)
        
        # Test documents
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Python is used for web development and data science.",
            "Deep learning uses neural networks with multiple layers.",
            "JavaScript is primarily used for web development.",
            "Natural language processing helps computers understand human language."
        ]
        
        # Test query
        query = "What is artificial intelligence and machine learning?"
        
        print(f"Model name: {provider.model_name}")
        
        # Test reranking
        print("\nTesting document reranking...")
        reranked_results = await provider.reranking_result(query, documents)
        
        print(f"Reranked {len(reranked_results)} documents:")
        for i, (doc, score) in enumerate(reranked_results, 1):
            print(f"{i}. Score: {score:.4f} - {doc[:60]}...")
        
        # Test top-k reranking
        print("\nTesting top-3 reranking...")
        top_3_results = await provider.rerank_top_k(query, documents, top_k=3)
        
        print(f"Top 3 results:")
        for i, (doc, score) in enumerate(top_3_results, 1):
            print(f"{i}. Score: {score:.4f} - {doc[:60]}...")
        
        print("✅ Reranker provider test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Reranker provider test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Testing Qwen3 model implementations...\n")
    
    embedding_success = await test_embedding_provider()
    reranker_success = await test_reranker_provider()
    
    if embedding_success and reranker_success:
        print("\n🎉 All tests completed successfully!")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)