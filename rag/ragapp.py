# rag/ragapp.py

from rag.data_loader import process_all_pdfs, split_documents
from rag.vectorstore import VectorStore
from rag.rag_retriever import RAGRetriever
from rag.rag import rag_advance
from rag.embeddings import EmbeddingManager


retriever = None   # GLOBAL retriever


def update_kb(DATA_DIR):

    global retriever

    all_docs = process_all_pdfs(DATA_DIR)

    chunks = split_documents(all_docs)
    texts = [doc.page_content for doc in chunks]

    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.generate_embeddings(texts)

    vectorstore = VectorStore()
    vectorstore.add_documents(chunks, embeddings)

    retriever = RAGRetriever(vectorstore, embedding_manager)

    return retriever


def rag_query(query: str):

    global retriever

    if retriever is None:
        raise ValueError(
            "Retriever not initialized. Call update_kb(DATA_DIR) first."
        )

    result = rag_advance(
        query,
        retriever,
        top_k=3,
        return_context=True
    )

    return result