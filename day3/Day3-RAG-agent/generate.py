### Generate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llmLoader import llm

system = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If there is a reference URL in the context, include it in your answer.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: {question}\n\n context: {context} "),
    ]
)

# Chain
rag_chain = prompt | llm | StrOutputParser()
