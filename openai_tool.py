"""
LangChain을 이용한 Tool 선택 예제
- SerpAPI Tool: 검색 도구
- Calculator Tool: 수학 계산 도구
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import tool

# 환경 변수 로드
load_dotenv()

# LLM 초기화 (OpenAI Function Calling 지원)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 도구 리스트
tools = []

# 1. 검색 도구 정의 (선택사항 - API 키가 있을 경우만)
serpapi_key = os.getenv("SERPAPI_API_KEY")
if serpapi_key:
    search = SerpAPIWrapper()
    
    @tool
    def search_web(query: str) -> str:
        """웹에서 정보를 검색합니다. 최신 정보, 날씨, 뉴스, 인구 등을 찾을 때 사용하세요.
        
        Args:
            query: 검색할 질문 또는 키워드
        """
        return search.run(query)
    
    tools.append(search_web)
    print("✅ 검색 도구(SerpAPI) 활성화됨")
else:
    print("⚠️  SERPAPI_API_KEY가 설정되지 않았습니다. 검색 도구를 사용할 수 없습니다.")
    print("   .env 파일에 SERPAPI_API_KEY를 추가하거나 환경 변수로 설정하세요.")

# 2. 계산 도구 정의 (항상 사용 가능)
@tool
def calculate(expression: str) -> str:
    """수학 계산을 수행합니다. 사칙연산, 제곱근, 거듭제곱 등을 계산할 수 있습니다.
    
    Args:
        expression: 계산할 수학 표현식 (예: "sqrt(25) + 10", "100 * 2")
    """
    try:
        # numexpr을 사용하여 안전하게 계산
        import numexpr as ne
        result = ne.evaluate(expression)
        return str(result)
    except Exception as e:
        return f"계산 오류: {e}"

tools.append(calculate)
print("✅ 계산 도구(Calculator) 활성화됨")

if not tools:
    print("\n❌ 사용 가능한 도구가 없습니다. 프로그램을 종료합니다.")
    exit(1)

# LLM에 도구 바인딩 (Function Calling)
llm_with_tools = llm.bind_tools(tools)

# Agent 실행 함수
def run_agent(question: str):
    """Agent를 실행하여 질문에 답합니다."""
    print(f"\n질문: {question}")
    print("-" * 70)
    
    messages = [{"role": "user", "content": question}]
    max_iterations = 10
    
    for i in range(max_iterations):
        print(f"\n[반복 {i+1}]")
        
        # LLM 호출
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Tool Call이 있는지 확인
        if not response.tool_calls:
            # 최종 답변
            print(f"✅ 최종 답변: {response.content}")
            return response.content
        
        # Tool 실행
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"🔧 도구 사용: {tool_name}")
            print(f"   입력: {tool_args}")
            
            # 해당 도구 찾기 및 실행
            selected_tool = {tool.name: tool for tool in tools}[tool_name]
            tool_result = selected_tool.invoke(tool_args)
            
            print(f"   결과: {tool_result[:200]}..." if len(str(tool_result)) > 200 else f"   결과: {tool_result}")
            
            # 도구 실행 결과를 메시지에 추가
            messages.append({
                "role": "tool",
                "content": str(tool_result),
                "tool_call_id": tool_call["id"]
            })
    
    return "최대 반복 횟수에 도달했습니다."

# 실행 예제
if __name__ == "__main__":
    print("=" * 70)
    print("  LangChain Tool 선택 예제 (OpenAI Function Calling)")
    print("=" * 70)
    
    # 예제 1: 계산만 사용
    print("\n\n[예제 1] 수학 계산")
    print("=" * 70)
    run_agent("25의 제곱근에 10을 더하면?")
    
    # 예제 2: 검색과 계산을 함께 사용
    print("\n\n[예제 2] 검색 + 계산")
    print("=" * 70)
    run_agent("서울의 현재 인구가 몇 명인지 검색하고, 그 숫자에 2를 곱한 값은?")