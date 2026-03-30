from langchain_ollama.llms import OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate
from embed import vectorstore
import os
import store

config = store.load_config()
retrieval_k = int(config.get("retrieval_k", 6))
max_context_chars = int(config.get("max_context_chars", 3500))
max_doc_chars = int(config.get("max_doc_chars", 700))

model = Ollama(
    model=config.get("default_model", "deepseek-coder-v2"),
    num_predict=int(config.get("num_predict", 192)),
    num_ctx=int(config.get("num_ctx", 2048)),
    num_thread=int(config.get("num_thread", max(1, (os.cpu_count() or 4) - 1))),
    temperature=float(config.get("temperature", 0.1)),
)

retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_k})
template = """
You are an assistant for answering questions based on the following ingested documents. 
Use the information in the documents to answer the question as best as you can. 
If you don't know the answer, say you don't know.
Always use the information in the documents and never make up an answer.
Here are some relevant docs: {reviews}
Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


def _build_bounded_context(docs, max_total_chars: int = 6000, max_doc_chars: int = 1200) -> str:
    lines = []
    total = 0
    for doc in docs:
        doc_text = doc.page_content.strip()
        if len(doc_text) > max_doc_chars:
            doc_text = doc_text[:max_doc_chars].rsplit(" ", 1)[0].strip() or doc_text[:max_doc_chars]
        line = f"[ID: {doc.id}] {doc_text}"
        if total + len(line) > max_total_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)

def askai(question):
    rag = retriever.invoke(question)
    rag_text = _build_bounded_context(
        rag,
        max_total_chars=max_context_chars,
        max_doc_chars=max_doc_chars,
    )
    result = chain.invoke({"reviews": rag_text, "question": question})
    print(result, "\n\n")

if __name__ == "__main__":
    while True:
        question = input("Ask a question (or type 'exit'): ")
        if question.lower() == "exit":
            break
        askai(question)