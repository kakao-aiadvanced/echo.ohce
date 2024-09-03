from fastapi import FastAPI, Request  # 수정된 부분
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from documentLoader import getContext
from hallucinationChecker import hallucinationChecker
from llmLoader import llm
from relevanceChecker import relevanceChecker

app = FastAPI()

def createAnswer(query):
    context = getContext(query)
    
    if not relevanceChecker(context = context, query = query):
        return 'I can not find relevant document'
    
    prompt = hub.pull("rlm/rag-prompt")  # 프롬폼트 템플릿을 가져온다. hub
    rag_chain = (
        {"context": lambda _: context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    attempt = 0
    
    while attempt < 5:
        # RAG 체인 사용하여 답변 생성
        answer = rag_chain.invoke(query)
        # 할루시네이션 여부 확인
        if hallucinationChecker(answer, query):
            print(f"Attempt {attempt + 1}: Hallucination detected. Retrying...")
            attempt += 1
        else:
            # 할루시네이션이 아닌 경우 답변 반환
            return answer
        
    return "fail to answer"
    
    


# POST 요청을 처리하는 엔드포인트 정의
@app.post("/query/")
async def process_query(request: Request):
    # 요청에서 받은 JSON 데이터를 처리
    body = await request.json()
    query = body.get("query")

    # 쿼리를 처리하고 응답 생성
    answer = createAnswer(query)

    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
