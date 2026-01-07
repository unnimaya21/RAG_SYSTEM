from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def generate_answer(prompt):
    return llm.invoke(prompt).content
