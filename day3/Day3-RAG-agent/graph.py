from pprint import pprint
from typing import List

from langchain_core.documents import Document
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from answer import answer_grader
from documentLoader import retriever
from generate import rag_chain
from halluciantionGrader import hallucination_grader
from retrievalGrader import retrieval_grader
from router import question_router
from tavily import TavilyClient

### State
tavily = TavilyClient(api_key='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')

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: str
    documents: List[Document]  # Ensure this holds Document objects, not str


### Nodes

def retrieve(state: GraphState):
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    print(question)
    print(documents)
    return {"documents": documents}


def generate(state: GraphState):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    print(f"question:: {question}")
    print(f"docuements:: {documents}")

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"generation": generation}


def grade_documents(state: GraphState):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "Yes"
    for d in documents:
        print(d.page_content)
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
    
        grade = score.get("score", "no")
        
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            web_search = "No"
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            
    if len(filtered_docs) > 0:
        return {"documents": filtered_docs, "web_search": web_search}
    

def web_search(state: GraphState):
    print("---WEB SEARCH---")
    question = state["question"]

    print(f"question:: {question}")
    # Web search
    docs = tavily.search(query=question)['results']
    print(f"docs:: {docs}")
    # Initialize lists to store content and URLs
    contents = []
    # Extract content and URLs from search results
    for d in docs:
        url = d.get("url", "")
        content = d["content"]
        # Add "reference: {url}" before the content
        content_with_reference = f"reference: {url}\n{content}"
        
        contents.append(content_with_reference)
    
    # Join all contents into a single string
    web_results_content = "\n".join(contents)

    # Create a Document object with the combined content
    web_results_document = Document(page_content=web_results_content)

    # Add the document to the list of documents
    
    documents = []
    documents.append(web_results_document)
    state["documents"] = documents  # Correctly assign the list to state["documents"]

    print(state["documents"] )
    # Return both documents and URLs in the response
    return state


def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "retrieve"


def decide_to_generate(state: GraphState) -> str:
    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state.get("web_search", "Yes")

    if web_search == "Yes":
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # 수정된 부분: documents가 비어있는지 확인
    if not documents:
        print("---NO DOCUMENTS AVAILABLE TO CHECK HALLUCINATIONS---")
        return "not supported"

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score["score"]

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("websearch", web_search)
workflow.add_node("generate", generate)
