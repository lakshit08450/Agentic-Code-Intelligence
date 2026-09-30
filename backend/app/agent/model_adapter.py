"""
Model Adapter — Clean Interface for the actual ML Model.
This isolates the model logic from the rest of the backend pipeline.
The trained retrieval model will be integrated here by the teammate later.
"""

from typing import List, Tuple
from app.indexer.embedder import get_embedder
from app.storage.faiss_store import FAISSStore

class RetrievalModel:
    """
    Adapter for the retrieval ML model.
    Currently uses the FAISS CPU embedder for testing.
    The teammate will replace this implementation with the actual trained model.
    """
    
    def __init__(self, faiss_store: FAISSStore = None):
        self.faiss = faiss_store
        self.embedder = get_embedder()
        self.is_loaded = False

    def load(self):
        """
        Load the model weights into memory/VRAM.
        """
        self.is_loaded = True
        # Teammate will initialize actual model here
        pass

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Execute semantic search.
        
        Args:
            query (str): The natural language query.
            top_k (int): Number of results to return.
            
        Returns:
            List[Tuple[str, float]]: A list of (chunk_id, score)
        """
        if self.faiss is None:
            return []
            
        # Development implementation (FAISS)
        # Teammate will replace this with their model inference
        query_vec = self.embedder.embed_query(query)
        return self.faiss.search(query_vec, top_k=top_k)
