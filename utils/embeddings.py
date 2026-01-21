"""
Embedding model wrapper for LMStudio API.
"""

import requests
from typing import List


class LMStudioBgeM3Dense:
    """
    Wrapper for LMStudio's BGE-M3 dense embedding model.
    Compatible with OpenAI-style embeddings API.
    """
    
    def __init__(self, base_url: str, model: str):
        self.url = f"{base_url}/embeddings"
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "input": texts  # Must be raw strings
            }
        )
        response.raise_for_status()
        return [d["embedding"] for d in response.json()["data"]]
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        
        Args:
            text: Query text to embed
            
        Returns:
            Single embedding vector
        """
        return self.embed_documents([text])[0]