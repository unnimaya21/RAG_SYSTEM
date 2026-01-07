from pipelines.rag_pipeline import run_rag
from dotenv import load_dotenv
load_dotenv()
import os
print("OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))

def main():
    print("=== RAG SYSTEM STARTED ===")

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        answer = run_rag(query)
        print("\n🧠 Answer:", answer)


if __name__ == "__main__":
    main()
