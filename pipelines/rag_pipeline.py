from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.retrievers.multi_query import MultiQueryRetriever

# -----------------------------
# Configuration
# -----------------------------
PERSIST_DIRECTORY = "chroma_db"
TOP_K = 5

# -----------------------------
# Load Vector DB
# -----------------------------
def load_vectorstore():
    embeddings = OpenAIEmbeddings()

    vectordb = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )

    return vectordb


# -----------------------------
# Build Retriever (Multi-Query)
# -----------------------------
def build_retriever(vectordb):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    base_retriever = vectordb.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    return retriever


# -----------------------------
# Build RAG Chain
# -----------------------------
def build_rag_chain():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a factual assistant.
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""
    )

    return llm, prompt


# -----------------------------
# Run RAG
# -----------------------------
def run_rag(query: str):
    print("🔍 Query:", query)

    vectordb = load_vectorstore()
    retriever = build_retriever(vectordb)
    llm, prompt = build_rag_chain()

    # Retrieve documents
    docs = retriever.get_relevant_documents(query)

    print(f"📄 Retrieved documents: {len(docs)}")

    if not docs:
        return "I don't know"

    # Debug: show retrieved content
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Doc {i} ---")
        print(doc.page_content[:300])

    context = "\n\n".join(doc.page_content for doc in docs)

    chain_input = prompt.format_messages(
        context=context,
        question=query
    )

    response = llm.invoke(chain_input)

    return response.content
