from dotenv import load_dotenv
import os
import langchain
from langchain_openai import ChatOpenAI
# LangSmith 연동을 위한 traceable 핸들러를 가져옵니다.
from langsmith import traceable

# .env 파일 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

print(f"LangChain 버전: {langchain.__version__}")

# 추적하려는 함수에 @traceable 데코레이터 추가
@traceable # 이 함수 호출을 LangSmith에서 추적합니다.
def my_llm_call(prompt: str):
    """지정된 프롬프트로 LLM을 호출하는 함수"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(prompt)
    return response.content

# 함수 실행
result = my_llm_call("LangSmith에 대해 한 문장으로 설명해줘.")
print(result)


