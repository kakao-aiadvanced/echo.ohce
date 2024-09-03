from llmLoader import llm


def hallucinationChecker(answer, query):
    # 모델에게 질문할 텍스트 정의
    question = (
        f"Query: {query}\n"
        f"Answer: {answer}\n"
        "Is the answer factually accurate and relevant to the query? "
        "If the answer contains hallucination or is not relevant, answer 'hallucination'. "
        "If the answer is accurate and relevant, answer 'accurate'."
    )

    # 모델에 질문을 전달하여 답변 받기
    response = llm(question).content

    # 응답을 확인하여 할루시네이션 판단
    if "hallucination" in response.lower():
        return True  # 할루시네이션임
    elif "accurate" in response.lower():
        return False  # 할루시네이션이 아님
    else:
        return None  # 확실하지 않은 경우

