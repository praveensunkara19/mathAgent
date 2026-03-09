#rag_retriever.py

import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class RAGRetriever:
    """Handles query-based retriever from the vector store"""

    def __init__(self, vector_store, embeding_manager):
        """"initialize the retriever"""

        self.vector_store = vector_store
        self.embedding_manager = embeding_manager

    def retrieve(self, query: str, top_k: int = 3, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        print(f"Retriving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # chroma db uses cosine distance
                    similarity_score = 1 - distance  # Convert cosine distance to similarity

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })

                # Check AFTER LOOP
                if not retrieved_docs:
                    print("No documents passed similarity threshold.")

                return retrieved_docs

            print("No documents returned from vector store.")
            return []

        except Exception as e:
            print(f"Error during retrival: {e}")
            return []


# rag_retiver = RAGRetriver(vectorstore,embeding_manager)


    