def hybrid_multi_query_retrieve(
    question: str,
    queries: list[str],
    bm25,
    vector,
    k: int = 5
):
    results = []
    seen = set()

    for q in queries:
        bm25_docs = bm25.invoke(q)
        vector_docs = vector.invoke(q)

        for doc in bm25_docs + vector_docs:
            key = doc.page_content
            if key not in seen:
                seen.add(key)
                results.append(doc)

    return results[:k]
