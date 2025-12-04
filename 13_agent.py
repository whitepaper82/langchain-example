import os
from dotenv import load_dotenv

# LangChain 관련 임포트
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama

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
travily_module = importlib.import_module("13_travily_class")
TavilySearchAgent = travily_module.TavilySearchAgent

# 13_custom_tool.py 동적 임포트
custom_tool_module = importlib.import_module("13_custom_tool")
add_numbers = custom_tool_module.add_numbers
multiply_numbers = custom_tool_module.multiply_numbers

def main():
    # 환경 변수 로드
    load_dotenv()


    # 1. LLM 초기화 (Ollama 사용)
    llm = ChatOllama(
        base_url="http://localhost:11434",
        model="kimjk/llama3.2-korean"
    )
    print("✅ LLM(llama3.2-korean) 초기화 완료")

    # 2. 도구(Tools) 준비

    # (1) QnA 에이전트 도구
    # QnAAgent 인스턴스 생성
    #qna_agent_instance = QnAAgent()
    
    # QnA 기능을 Tool로 래핑
    #qna_tool = Tool(
    #    name="PDF_QnA",
    #    func=qna_agent_instance.answer,
    #    description="SPRi AI 산업동향 PDF 문서에 대한 질문에 답변할 때 사용합니다. 입력값은 질문 문자열입니다."
    #)

    # (2) Tavily 검색 에이전트 도구
    tavily_agent_instance = TavilySearchAgent()
    # TavilySearchAgent에서 제공하는 도구 가져오기
    search_tool = tavily_agent_instance.get_general_search_tool()
    # 도구 이름과 설명이 이미 설정되어 있지만, 필요하다면 수정 가능
    # search_tool.name = "Web_Search"
    # search_tool.description = "최신 정보나 웹 검색이 필요할 때 사용합니다."

    # (3) 커스텀 계산 도구
    # add_numbers, multiply_numbers는 이미 @tool 데코레이터로 정의됨

    # 모든 도구를 리스트로 통합
    tools = [
        #qna_tool,
        search_tool,
        add_numbers,
        multiply_numbers
    ]
    
    print(f"✅ 사용 가능한 도구: {[t.name for t in tools]}")

    # 3. 에이전트 초기화 (create_react_agent 사용)
    # initialize_agent는 deprecated 되었거나 import 에러가 발생하므로 create_react_agent 사용
    from langchain.agents import create_agent
    from langchain_classic import hub

    # ReAct 프롬프트 로드
    # hub.pull("hwchase17/react")를 사용하거나 직접 정의할 수 있습니다.
    # 여기서는 hub에서 가져오는 방식을 시도합니다.
    try:
        prompt = hub.pull("hwchase17/react")
    except Exception as e:
        print(f"⚠️ 프롬프트 로드 실패, 기본 프롬프트를 사용합니다: {e}")
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(
            """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
        )

    # 에이전트 생성
    agent = create_agent(llm, tools)

    # 에이전트 실행기 생성
    #agent_executor = AgentExecutor(
    #    agent=agent, 
    #    tools=tools, 
    #    verbose=True, 
    #    handle_parsing_errors=True
    #)
    #print("✅ 에이전트 생성 완료 (create_react_agent)")

    # 4. 에이전트 실행 테스트
    print("\n========== 에이전트 테스트 시작 ==========")
    
    # 시나리오 1: PDF 문서 관련 질문
    query1 = "SPRi AI Brief에서 말하는 구글의 최신 동영상 생성 AI 모델 이름은 뭐야?"
    print(f"\n[질문 1] {query1}")
    agent.invoke({"input": query1})

    # 시나리오 2: 웹 검색이 필요한 질문
    query2 = "현재 한국의 대통령은 누구야?"
    print(f"\n[질문 2] {query2}")
    agent.invoke({"input": query2})

    # 시나리오 3: 계산이 필요한 질문
    query3 = "123 더하기 456은 몇이야? 그리고 그 결과에 2를 곱해줘."
    print(f"\n[질문 3] {query3}")
    agent.invoke({"input": query3})

if __name__ == "__main__":
    main()
