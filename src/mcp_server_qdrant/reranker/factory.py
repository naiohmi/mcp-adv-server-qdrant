from mcp_server_qdrant.reranker.base import RerankProvider
from mcp_server_qdrant.reranker.types import RerankerProviderType


def create_reranker_provider(provider_type: RerankerProviderType, model_name: str) -> RerankProvider:
    """
    Create a reranker provider based on the specified type.
    :param provider_type: The type of reranker provider.
    :param model_name: The name of the model to use.
    :return: An instance of the specified reranker provider.
    """
    if provider_type == RerankerProviderType.TRANSFORMERS:
        from mcp_server_qdrant.reranker.transformers import TransformersProvider

        return TransformersProvider(model_name)
    else:
        raise ValueError(f"Unsupported reranker provider: {provider_type}")
