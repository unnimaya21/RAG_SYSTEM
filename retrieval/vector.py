from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def vector_retriever():
    vectordb = Chroma(
        persist_directory="db",
        embedding_function=OpenAIEmbeddings()
    )
    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "lambda_mult": 0.5}
    )
