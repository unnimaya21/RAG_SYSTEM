from langchain_community.retrievers import BM25Retriever

def bm25_retriever(docs, k=5):
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever
