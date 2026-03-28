from langchain_ollama.llms import OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate
from embed import retriever

model = Ollama(model="llama3.2")

template = """
You are a helpful assistant that answers questions using documentation.
Use only the context below to answer. If you don't know, say so.

Context:
{reviews}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def askai(question):
    rag = retriever.invoke(question)
    rag_text = "\n".join(f"[{doc.metadata['source']}]\n{doc.page_content}" for doc in rag)
    result = chain.invoke({"reviews": rag_text, "question": question})
    print(result, "\n\n")

if __name__ == "__main__":
    while True:
        question = input("Ask a question (or type 'exit'): ")
        if question.lower() == "exit":
            break
        askai(question)