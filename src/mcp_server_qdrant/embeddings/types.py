from enum import Enum


class EmbeddingProviderType(Enum):
    FASTEMBED = "fastembed"
    TRANSFORMERS = "transformers"
