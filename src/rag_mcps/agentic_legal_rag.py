from dotenv import load_dotenv

load_dotenv()

from typing import Annotated, Sequence, Literal
from langgraph.graph.message import add_messages

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict

# -----------------------------
# 🧠 1. Vector Store & Embedding
# -----------------------------
persist_dir = "/workspace/data/chromaDB"
ENCODER = "dragonkue/BGE-m3-ko"

embedding_model = HuggingFaceEmbeddings(model_name=ENCODER)
vectorstore = Chroma(
    persist_directory=persist_dir,
    collection_name="legal_docs_1",
    embedding_function=embedding_model,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


# -----------------------------
# 🧾 2. Tool 정의
# -----------------------------
@tool
def retrieve_safety_legal_docs(query: str) -> str:
    """
    Search and return legal documents or regulations related to safety rules on construction sites.
    Returns the document title and content.
    """
    docs: list[Document] = retriever.invoke(query)
    result = "\n\n".join(
        f"Title: {doc.metadata.get('title', 'No Title')}\n{doc.page_content}"
        for doc in docs
    )
    return result


tools = [retrieve_safety_legal_docs]


# -----------------------------
# 🧩 3. State 정의
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# -----------------------------
# 🧠 4. 노드 정의
# -----------------------------
def supervisor(state):
    print("---CALL SUPERVISOR---")
    messages = state["messages"]

    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")
    model = model.bind_tools(tools)

    response = model.invoke(messages)

    if response.tool_calls:
        new_messages = messages + [response]

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            # Tool 찾기
            tool = next((t for t in tools if t.name == tool_name), None)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found.")

            # ✅ invoke()로 실행
            result = tool.invoke(tool_args)

            # ToolMessage 추가
            tool_msg = ToolMessage(content=result, tool_call_id=tool_call_id)
            new_messages.append(tool_msg)

        return {"messages": new_messages}

    return {"messages": messages + [response]}


def RAG_retrieve(state):
    print("---RAG QUERY + RETRIEVE---")

    question = state["messages"][0].content

    # 쿼리 생성
    query_prompt = PromptTemplate(
        template="""You are a query expert. Based on the user question below,
        formulate a search query that would retrieve documents relevant to it.
        User question: {question}
        Search query: """,
        input_variables=["question"]
    )
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    query_chain = query_prompt | model | StrOutputParser()
    search_query = query_chain.invoke({"question": question})

    print(f"[Generated Query]: {search_query}")

    # 문서 검색
    docs: list[Document] = retriever.invoke(search_query)
    result = "\n\n".join(
        f"Title: {doc.metadata.get('title', 'No Title')}\n{doc.page_content}"
        for doc in docs
    )
    return {"messages": state["messages"] + [AIMessage(content=result)]}


def rewrite(state):
    print("---REWRITE QUERY---")
    question = state["messages"][0].content
    msg = [
        HumanMessage(
            content=f"""Look at the input and try to reason about the underlying semantic intent.\n
Here is the initial question:\n{question}\n
Formulate an improved question:"""
        )
    ]
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    print("---FINAL GENERATE---")
    question = state["messages"][0].content
    context = state["messages"][-1].content

    #prompt = hub.pull("rlm/rag-prompt")
    prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a legal assistant. Based on the user question below,
    summarize the following legal texts in a way that directly answers the user's intent.
    User question: {question}
    
    Legal documents: {context}
    
    Summary:""")
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)
    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({"context": context, "question": question})
    return {"messages": [response]}


# -----------------------------
# 🔀 5. 조건자 정의
# -----------------------------
def tools_condition(state) -> Literal["RAG_retrieve", END]:
    messages = state["messages"]

    # AIMessage와 그에 대한 ToolMessage 응답이 모두 있는 경우 → RAG_retrieve
    for i in range(len(messages) - 1):
        msg = messages[i]
        next_msg = messages[i + 1]

        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_call_ids = {call["id"] for call in msg.tool_calls}
            if isinstance(next_msg, ToolMessage) and next_msg.tool_call_id in tool_call_ids:
                return "RAG_retrieve"

    return END


def grade_documents(state) -> Literal["generate", "rewrite"]:
    print("---CHECK RAG QUALITY---")

    class Grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    llm_with_output = model.with_structured_output(Grade)

    prompt = PromptTemplate(
        template = """You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_output
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content

    result = chain.invoke({"question": question, "context": docs})
    return "generate" if result.binary_score == "yes" else "rewrite"


# -----------------------------
# 🧠 6. LangGraph 워크플로우 정의
# -----------------------------
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor)
workflow.add_node("RAG_retrieve", RAG_retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    tools_condition,
    {
        "RAG_retrieve": "RAG_retrieve",
        "generate": "generate",
        "rewrite": "rewrite",
        END: END,
    }
)

workflow.add_conditional_edges(
    "RAG_retrieve",
    grade_documents,
    {
        "generate": "generate",
        "rewrite": "rewrite",
    }
)

workflow.add_edge("rewrite", "supervisor")
workflow.add_edge("generate", END)

graph = workflow.compile()

def construction_safety_legal_docs_RAG(query: str) -> str:
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }

    output_log = []

    for step in graph.stream(initial_state):
        for node_name, value in step.items():
            output_log.append(f"[Node: {node_name}]\n")
            # messages가 list일 경우 예쁘게 출력
            if isinstance(value, dict) and "messages" in value:
                for msg in value["messages"]:
                    if hasattr(msg, 'type') and hasattr(msg, 'content'):
                        output_log.append(f"{msg.type.upper()}: {msg.content}\n")
                    else:
                        output_log.append(f"{msg}\n")
            else:
                output_log.append(f"{value}\n")

    return "".join(output_log)

if __name__ == "__main__":
    query = "건설 현장에서 굴착 작업을 할 때 사람이 접근하는 것을 통제해야 하는 법적 근거가 필요해."

    result = construction_safety_legal_docs_RAG(query)
    print(result)