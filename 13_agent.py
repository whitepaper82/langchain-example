import os
from dotenv import load_dotenv

# LangChain 관련 임포트
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

# 우리가 만든 모듈 임포트
# 1. QnA Class (12_qna_class.py)
# 주의: 파일명이 숫자로 시작하므로 importlib을 사용하거나, 같은 디렉토리라면 그냥 import 가능하지만
# 파이썬 변수명 규칙상 숫자로 시작하는 모듈은 import 문으로 직접 가져오기 까다로울 수 있습니다.
# 하지만 여기서는 일반적인 import가 동작한다고 가정하고 시도해봅니다.
# 만약 import 에러가 나면 importlib을 사용해야 합니다.
# 보통 숫자로 시작하는 파일은 import 12_qna_class 가 안되므로, 
# from ... import ... 구문도 문제가 될 수 있습니다.
# 일단 importlib을 사용하여 안전하게 가져오겠습니다.
import importlib

# 12_qna_class.py 동적 임포트
qna_module = importlib.import_module("12_qna_class")
QnAAgent = qna_module.QnAAgent

# 13_travily_class.py 동적 임포트
# 13_travily_class.py 동적 임포트
travily_module = importlib.import_module("13_travily_class")
TavilySearchAgent = travily_module.TavilySearchAgent

#custom_tool_module = importlib.import_module("13_custom_tool")
custom_tool_module = importlib.import_module("13_custom_tool")
add_numbers = custom_tool_module.add_numbers
multiply_numbers = custom_tool_module.multiply_numbers

#from langchain_core.tools import tool
#@tool
#def add_numbers(a: int, b: int) -> int:
#    """Add two numbers"""
#    return a + b
#
#@tool
#def multiply_numbers(a: int, b: int) -> int:
#    """Multiply two numbers"""
#    return a * b

# Helper function to print answer and tool usage
def print_result_with_tool_usage(result):
    from langchain_core.messages import AIMessage
    
    # 1. Collect used tools
    used_tools = []
    # 결과가 dict이고 messages 키가 있는 경우 (LangGraph agent)
    if isinstance(result, dict) and "messages" in result:
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    used_tools.append(tool_call['name'])
        final_content = result["messages"][-1].content
    else:
        # Fallback
        final_content = str(result)
    
    if used_tools:
        print(f"🛠️  사용된 도구: {', '.join(used_tools)}")
    else:
        print("🛠️  사용된 도구: 없음")

    # 2. Print final answer
    print("답변:", final_content)

def main():
    # 환경 변수 로드
    load_dotenv()

    # LangSmith 모니터링 설정
    # .env 파일에 LANGCHAIN_API_KEY가 있어야 합니다.
    # 만약 환경변수가 설정되어 있지 않다면 강제로 설정합니다.
    if os.getenv("LANGCHAIN_TRACING_V2") is None:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
    # 프로젝트 이름은 원하는 이름으로 설정하세요 on LangSmith >> Projects
    os.environ["LANGCHAIN_PROJECT"] = "Ollama-Agent-Monitoring"

    # API Key 확인
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("⚠️  WARNING: LANGCHAIN_API_KEY가 설정되지 않았습니다. LangSmith 모니터링이 작동하지 않을 수 있습니다.")
    else:
        print(f"✅ LangSmith Tracing Enabled (Project: {os.getenv('LANGCHAIN_PROJECT')})")


    # 1. LLM 초기화 (Ollama 사용)
    llm = ChatOllama(
        base_url="http://localhost:11434",
        model="kimjk/llama3.2-korean"
    )
    print("✅ LLM(llama3.2-korean) 초기화 완료")

    # 2. 도구(Tools) 준비

    # (1) QnA 에이전트 도구
    # QnAAgent 인스턴스 생성
    qna_agent_instance = QnAAgent()
    
    # QnA 기능을 Tool로 래핑
    qna_tool = Tool(
        name="SPRi_QA",
        func=qna_agent_instance.answer,
        description="SPRi AI Brief 관련 질문에 대해서는 반드시 이 도구를 사용해야 합니다. 입력값은 질문 문자열입니다.",
        return_direct=True
    )

    # (2) Tavily 검색 에이전트 도구
    tavily_agent_instance = TavilySearchAgent()
    # TavilySearchAgent에서 제공하는 도구 가져오기
    search_tool = tavily_agent_instance.get_general_search_tool()
    # 도구 이름과 설명이 이미 설정되어 있지만, 필요하다면 수정 가능
    search_tool.name = "Web_Search"
    search_tool.description = "웹 검색이 필요한 질문에 대해서는 반드시 이 도구를 사용해야 합니다."

    # (3) 커스텀 계산 도구
    # add_numbers, multiply_numbers는 이미 @tool 데코레이터로 정의됨

    # 모든 도구를 리스트로 통합
    tools = [
        qna_tool,
        search_tool,
        add_numbers,
        multiply_numbers
    ]
    
    print(f"✅ 사용 가능한 도구: {[t.name for t in tools]}")

    # 3. 에이전트 초기화 (create_agent 사용)
    # 시스템 프롬프트 정의: 루프 방지 및 도구 사용 가이드
    system_prompt = (
        "당신은 유능한 AI 어시스턴트입니다. 질문에 답하기 위해 사용 가능한 도구를 활용하세요. "
        "도구가 반환한 정보를 바탕으로 답변을 작성하세요. "
        "만약 도구에서 유용한 정보를 얻지 못했다면, 솔직하게 모른다고 대답하거나 대안을 제시하세요. "
        "절대로 동일한 입력으로 같은 도구를 반복해서 호출하지 마세요. "
        "최종 답변은 한국어로 작성해 주세요."
    )
    
    # 에이전트 생성 (LangGraph 기반)
    agent = create_agent(llm, tools, system_prompt=system_prompt)

    print("✅ 에이전트 생성 완료 (create_agent + System Prompt)")

    # 4. 에이전트 실행 테스트
    print("\n========== 에이전트 테스트 시작 ==========")
    
    # 시나리오 1: PDF 문서 관련 질문
    query1 = "SPRi AI Brief 문서 내용 중 구글의 최신 동영상 생성 AI 모델 이름은 뭐야?"
    print(f"\n[질문 1] {query1}")
    result = agent.invoke({"messages": [HumanMessage(content=query1)]})
    print_result_with_tool_usage(result)

    # 시나리오 2: 웹 검색이 필요한 질문
    query2 = "현재 한국의 대통령이 누구인지 웹 검색을 해서 알려주세요"
    print(f"\n[질문 2] {query2}")
    result = agent.invoke({"messages": [HumanMessage(content=query2)]})
    print_result_with_tool_usage(result)

    # 시나리오 3: 계산이 필요한 질문
    query3 = "123 더하기 456은 몇이야? 그리고 그 결과에 2를 곱해줘."
    print(f"\n[질문 3] {query3}")
    result = agent.invoke({"messages": [HumanMessage(content=query3)]})
    print_result_with_tool_usage(result)

if __name__ == "__main__":
    main()
