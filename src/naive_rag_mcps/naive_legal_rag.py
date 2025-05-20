from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

persist_dir = "/workspace/data/chromaDB"
ENCODER = "dragonkue/BGE-m3-ko"

embedding_model = HuggingFaceEmbeddings(model_name=ENCODER)
vectorstore = Chroma(
    persist_directory=persist_dir,
    collection_name="legal_docs_1",
    embedding_function=embedding_model,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

def construction_safety_law_RAG(query: str) -> str:
    retrieved_docs = retriever.invoke(query)
    result = "\n\n".join(
        f"Title: {doc.metadata.get('title', 'No Title')}\n{doc.page_content}"
        for doc in retrieved_docs
    )

    return result

if __name__ == "__main__":
    query = "건설 현장에서 굴착 작업을 할 때 사람이 접근하는 것을 통제해야 하는 법적 근거가 필요해."

    result = construction_safety_law_RAG(query)
    print(result)
