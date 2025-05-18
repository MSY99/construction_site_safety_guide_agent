from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Legal_Docs_RAG")

persist_dir = "/workspace/data/chromaDB"
ENCODER = "dragonkue/BGE-m3-ko"

embedding_model = HuggingFaceEmbeddings(model_name=ENCODER)
vectorstore = Chroma(
    persist_directory=persist_dir,
    collection_name="legal_docs_1",
    embedding_function=embedding_model,
)

retriever = vectorstore.as_retriever()

@mcp.tool()
def construction_safety_law_RAG(query: str) -> str:
    retrieved_docs = retriever.invoke(query)
    result = "\n\n".join(
        f"Title: {doc.metadata.get('title', 'No Title')}\n{doc.page_content}"
        for doc in retrieved_docs
    )

    return result

if __name__ == "__main__":
    mcp.run(transport="stdio")
