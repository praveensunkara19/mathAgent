#embeddings.py

import numpy as np
from sentence_transformers import SentenceTransformer

from typing import List, Dict, Any, Tuple

    
class EmbeddingManager:
        """Handles document embedding generation using SentanceTransformer"""

        def __init__(self, model_name: str="all-mpnet-base-v2"):
            """ Initialise the embeddings manager"""

            self.model_name = model_name
            self.model = None
            self._load_model()

        def _load_model(self):
            """load the SentenceTransformer model"""

            try:
                print(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}") #768
            except Exception as e:
                print(f"Error loading model{self.model_name}:{e}")
                raise
        
        def generate_embeddings(self, texts: List[str]) -> np.array:
            """Generate embeddings for list of texts"""

            if not self.model:
                raise ValueError("Model not loaded")

            print(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.model.encode(texts, show_progress_bar = True)
            print(f"generated embeddings with shape:{embeddings.shape}")
            return embeddings


