from dotenv import load_dotenv
import os
import langchain
from langchain_openai import ChatOpenAI
# LangSmith 연동을 위한 traceable 핸들러를 가져옵니다.
from langsmith import traceable

# .env 파일 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# LangSmith 추적을 위한 환경 변수 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "langchain-tutorial"

print(f"LangChain 버전: {langchain.__version__}")

# 간단한 LangChain 예제 (Agent 없이)
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 검색 도구 초기화
search = SerpAPIWrapper()

@traceable  # 검색 함수를 LangSmith에서 추적합니다.
def search_weather(query: str):
    """날씨 정보를 검색하는 함수"""
    return search.run(query)

@traceable  # 계산 함수를 LangSmith에서 추적합니다.
def calculate_temperature(base_temp: int, add_temp: int):
    """온도를 계산하는 함수"""
    return base_temp + add_temp

@traceable  # 메인 처리 함수를 LangSmith에서 추적합니다.
def process_weather_request():
    """날씨 검색 및 계산을 처리하는 메인 함수"""
    # 1. 날씨 검색
    weather_query = "내일 서울 날씨"
    weather_result = search_weather(weather_query)
    print(f"검색 결과: {weather_result}")
    
    # 2. 간단한 계산 (예시로 5도 더하기)
    base_temp = 15  # 예시 온도
    add_temp = 5
    final_temp = calculate_temperature(base_temp, add_temp)
    
    # 3. LLM을 사용한 최종 답변 생성
    prompt = ChatPromptTemplate.from_template("""
    다음 정보를 바탕으로 답변해주세요:
    검색 결과: {weather_info}
    기본 온도: {base_temp}도
    추가 온도: {add_temp}도
    최종 온도: {final_temp}도
    
    내일 서울의 날씨에 대해 간단히 설명하고, {base_temp}도에 {add_temp}도를 더하면 {final_temp}도가 된다고 답변해주세요.
    """)
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "weather_info": weather_result,
        "base_temp": base_temp,
        "add_temp": add_temp,
        "final_temp": final_temp
    })
    
    return result

# 함수 실행
print("=== LangChain @traceable 데모 ===")
result = process_weather_request()
print(f"\n최종 결과: {result}")