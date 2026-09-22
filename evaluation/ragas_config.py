import os
import google.api_core.exceptions
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)


@retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=60),
        retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted) | retry_if_exception_type(google.api_core.exceptions.ServiceUnavailable)
)
def get_ragas_llm(model_name: str = "gemini-2.5-pro"):
    return LangchainLLMWrapper(
        ChatVertexAI(model_name=model_name, temperature=0, project=os.environ["GOOGLE_CLOUD_PROJECT"])
    )

def get_ragas_embeddings():
    return LangchainEmbeddingsWrapper(
        VertexAIEmbeddings(model_name="text-embedding-004", project=os.environ["GOOGLE_CLOUD_PROJECT"])
    )
