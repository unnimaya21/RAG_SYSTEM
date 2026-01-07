from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def generate_queries(question: str, n: int = 3) -> list[str]:
    prompt = f"""
Generate {n} different search queries that capture
different ways the following question might be phrased.

Question: {question}

Return each query on a new line.
"""

    response = llm.invoke(prompt).content

    queries = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]
    return list(set(queries))  # deduplicate
