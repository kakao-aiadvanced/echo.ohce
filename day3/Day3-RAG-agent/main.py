from pprint import pprint

from fastapi import FastAPI, Request  # 수정된 부분

from graph import workflow, decide_to_generate, grade_generation_v_documents_and_question, retrieve, grade_documents

app = FastAPI()


workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)

workflow.add_edge("websearch", "grade_documents")


workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": "__end__",
        "not useful": "websearch",
    }
)




def createAnswer(query):
    # Compile
    # Run the workflow with the provided inputs and collect outputs
    app = workflow.compile()
    inputs = {"question": query}


    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")

            # 수정된 부분: value가 None인지 먼저 확인
            if value is not None:
                if "generation" in value:
                    return value["generation"]  # Return the generation if found
                elif isinstance(value, str):  # Check if the value is a string
                    return value  # Return the string value directly if that's the output
            else:
                print(f"Warning: Value for key '{key}' is None.")  # 디버깅용 출력

    return "I don't know"



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
