from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

def get_ragas_llm(model_name: str = "gemini-2.5-pro"):
    return LangchainLLMWrapper(
        ChatVertexAI(model_name=model_name, temperature=0)
    )

def get_ragas_embeddings():
    return LangchainEmbeddingsWrapper(
        VertexAIEmbeddings(model_name="text-embedding-004")
    )
