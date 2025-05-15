from dotenv import load_dotenv

from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
import pprint
import uuid

from langchain import hub
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode

load_dotenv()

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Safety_Cases_RAG")

persist_dir = "/workspace/data/chromaDB"

ENCODER = "dragonkue/BGE-m3-ko"
embedding_model = HuggingFaceEmbeddings(
    model_name=ENCODER,
)

vectorstore = Chroma(
    persist_directory=persist_dir,
    collection_name="safety_case_1",
    embedding_function=embedding_model,
)
retriever = vectorstore.as_retriever()

@tool
def retrieve_safety_cases_docs(query: str) -> str:
    """
    Search and return some safety cases on construction sites.
    Returns the document title and content.
    """
    docs: list[Document] = retriever.invoke(query)
    result = "\n\n".join(
        f"""Case_factor: {doc.metadata.get('case_factor', 'No case_factor')}, Case_category: {doc.metadata.get('case_category', 'No case_category')}\n
        {doc.page_content}"""
        for doc in docs
    )
    return result

tools = [retrieve_safety_cases_docs]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state
    
    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---Check Relevance---")

    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    model = ChatOpenAI(temperature = 0, model = "gpt-4o-mini", streaming = True)
    llm_with_tool = model.with_structured_output(grade)

    prompt = PromptTemplate(
        template = """You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_tool
    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"
    
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"
    
def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.
    
    Args:
        state (messages): The current state
    
    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")
    model = model.bind_tools(tools)
    response = model.invoke(messages)

    return {"messages": [response]}

# 노드: generate_query
def generate_query(state):
    print("--GENERATE QUERY FOR RETRIEVER---")
    user_question = state["messages"][0].content

    prompt = PromptTemplate(
        template="""You are a query expert. Based on the user question below,
        formulate a search query that would retrieve documents relevant to it.
        User question: {question}
        Search query: """,
        input_variables=["question"]
    )

    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    chain = prompt | model | StrOutputParser()

    search_query = chain.invoke({"question": user_question})
    print(f"[Generated Query]: {search_query}")

    tool_call = ToolCall(
        id=str(uuid.uuid4()),  # ✅ 필수
        name="retrieve_safety_legal_docs",
        args={"query": search_query}
    )
    ai_msg = AIMessage(content="", tool_calls=[tool_call])

    return {
        "messages": state["messages"] + [ai_msg]
    }

# 노드: rewrite (질문 개선)
def rewrite(state):
    print("---TRANSFORM QUERY---")
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

# 노드: generate (최종 응답 생성)
def generate(state):
    print("---GENERATE---")
    question = state["messages"][0].content
    context = state["messages"][-1].content

    #prompt = hub.pull("rlm/rag-prompt")
    #prompt = "You are an expert who can summarize legal texts. Please summarize these legal pharagraphs."
    prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a construction site's safety guard. Based on the user question below,
    summarize the following legal texts in a way that directly answers the user's intent.
    User question: {question}
    
    Legal documents: {context}
    
    Summary:""")
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)
    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({"context": context, "question": question})
    return {"messages": [response]}

# 워크플로우 구성
workflow = StateGraph(AgentState)

# 노드 등록
workflow.add_node("agent", agent)
workflow.add_node("generate_query", generate_query)
workflow.add_node("retrieve", ToolNode([retrieve_safety_cases_docs]))
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

# 엣지 연결
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "generate_query",
        END: END,
    }
)
workflow.add_edge("generate_query", "retrieve")
workflow.add_conditional_edges("retrieve", grade_documents, {
    "generate": "generate",
    "rewrite": "rewrite",
})
workflow.add_edge("rewrite", "agent")
workflow.add_edge("generate", END)

graph = workflow.compile()

'''inputs = {
    "messages": [
        ("user", "굴착 작업 중에 부딪힘 사고 사례를 알려줘."),
    ]
}

for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from the node '{key}': ")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width = 80, depth=None)
    pprint.pprint("\n---\n")'''

@mcp.tool()
def construction_safety_cases_RAG(query: str) -> str:
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }

    final_state = graph.invoke(initial_state)
    messages = final_state["messages"][-1].content

    return str(messages)

if __name__ == "__main__":
    # stdio 전송을 사용하여 서버 실행
    print("Starting Safety cases RAG MCP server via stdio...")
    mcp.run(transport="stdio")