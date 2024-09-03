import string

from llmLoader import llm


def relevanceChecker(context, query):
    # 모델에게 질문할 텍스트 정의
    question = (
        f"Context: {context}\n"
        f"Query: {query}\n"
        "Is the query relevant to the context? If it is relevant, answer 'yes'. If it is not relevant, answer 'no'."
    )

    # 모델에 질문을 전달하여 답변 받기
    response = llm(question).content.lower().strip().rstrip(string.punctuation)
    
    print("aa")
    print(response)
    
    
    # 응답을 확인하여 관련성 판단
    if "yes" in response:
        return True
    elif "no" in response:
        return False
    else:
        return None  # 확실하지 않은 경우

