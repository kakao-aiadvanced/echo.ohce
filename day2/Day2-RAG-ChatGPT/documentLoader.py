import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llmLoader import llm

# Load, chunk and index the contents of the blog.

# 여러 URL 리스트 정의
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 빈 리스트로 초기화하여 모든 문서들을 저장
all_docs = []

# 각 URL에 대해 데이터를 로드하고 처리
for url in urls:
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    all_docs.extend(docs)  # 모든 문서를 하나의 리스트에 추가

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)

# 벡터스토어 생성 및 임베딩 저장
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

print("모든 URL의 데이터가 벡터 DB에 성공적으로 저장되었습니다.")

# Retrieve and generate using the relevant snippets of the blog.
# retrieval 을 수행할수 있는 retriever 객체를 생성 
retriever = vectorstore.as_retriever()

def format_docs(docs):
    print(f"docs: {docs}")
    return "\n\n".join(doc.page_content for doc in docs)


def getContext(query):
    return (retriever | format_docs).invoke(query)



print(getContext("What is Task Decomposition?"))


# rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
# )
# 
# response = rag_chain.invoke("What is Task Decomposition?")
