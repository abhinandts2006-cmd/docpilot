from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

EMBEDDINGS = OllamaEmbeddings(model="mxbai-embed-large:335m")
DB_LOCATION = "./chroma_langchain_db"
COLLECTION = "docpilot"

vectorstore = Chroma(
    collection_name=COLLECTION,
    persist_directory=DB_LOCATION,
    embedding_function=EMBEDDINGS
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

model = OllamaLLM(model="llama3.2")

template = """
You are a helpful assistant that answers questions using documentation.
Use only the context below to answer. If you don't know, say so.

Context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question = input("\nAsk a question (or type 'exit'): ")
    if question.lower() == "exit":
        break
    docs = retriever.invoke(question)
    context = "\n\n".join(f"[{doc.metadata['source']}]\n{doc.page_content}" for doc in docs)
    print("\nSources used:")
    for doc in docs:
        print(f"  - {doc.metadata['source']}")
    print("\nAnswer:")
    result = chain.invoke({"context": context, "question": question})
    print(result)