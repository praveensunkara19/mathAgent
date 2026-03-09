# rag.py

from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

groq_api = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    api_key=groq_api,
    model_name="openai/gpt-oss-120b",
    temperature=0.1,
    max_tokens=1024
)




def rag_advance(query, retriever, top_k=5, min_score=0.0, return_context=False):
    """
    RAG pipeline with extra features like - Answer, Score, Confidence score, and optionally full context.
    """

    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    if not results:
        return {
            'answer': 'No relevant context found.',
            'sources': [],
            'confidence': 0.0,
            'context': '' if return_context else None
        }

    context = "\n\n".join([doc['content'] for doc in results])
    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', 'unknown'),
        'score': doc['similarity_score'],
        'preview': doc['content'][:300] + "..."
    } for doc in results]

    confidence = max([doc['similarity_score'] for doc in results])

    

    output = {
        # 'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }

    if return_context:
        output['context'] = context

    return output
