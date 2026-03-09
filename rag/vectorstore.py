#vectorstore.py

import numpy as np
import os
import chromadb
from chromadb.config import Settings
import uuid

from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
        """Manages document embeddings in a chromadb"""

        def __init__(self,collection_name: str="pdf_documents_db", persistant_directory: str="rag/data/vector_store"):

            self.collection_name = collection_name
            self.persistant_directory = persistant_directory
            self.Client = None
            self.collection = None
            self._initialize_store()

        def _initialize_store(self):
            """Initialize ChromaDB clinet and collection"""
            try:
                os.makedirs(self.persistant_directory, exist_ok=True)
                self.client = chromadb.PersistentClient(path=self.persistant_directory)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description":"PDF document embeddings for RAG"}
                    )
                print(f"Vector store initialized. collection: {self.collection_name}")
                print(f"Existing document in collection: {self.collection.count()}")


            except Exception as e:
                print(f"Error initializing the vectore store {e}")
                raise


        def add_documents(self, documents: List[Any], embeddings: np.ndarray):

            """Adding documents and their embeddings to vector store"""
             
            if len(documents) != len(embeddings):
                raise ValueError("No.of documents must matach no.of embeddings")
                
            print(f"Adding {len(documents)} documents to vectore store...")

                #preparing data for Chromadb
            ids = []
            metadatas = []
            documents_text = []
            embeddings_list = []

            for i, (doc, embeddings) in enumerate(zip(documents,embeddings)):
                #generate unique ID
                doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
                ids.append(doc_id)

                    #prepare metadata
                metadata = dict(doc.metadata)
                metadata['doc_index'] = i  #matching intial i index(0)  with staring page i.e 1
                metadata['content_length'] = len(doc.page_content)
                metadatas.append(metadata)

                #Document content
                documents_text.append(doc.page_content)

                
                #embeddings
                embeddings_list.append(embeddings.tolist())

            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                    documents=documents_text
                )
                print(f"Successfully added {len(documents)} to the vectorr store")
                print(f"total documents in the vector store: {self.collection.count()}")

            except Exception as e:
                print(f"error adding documents to the  vector")
                raise
                

# vectorstore = VectorStore()

# vectorstore