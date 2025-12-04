import os
from dotenv import load_dotenv

# LangChain 관련 임포트
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.agents.format_scratchpad import format_log_to_str

# 우리가 만든 모듈 임포트
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
    # native tool calling을 지원하지 않는 모델도 사용 가능 (예: gemma3:1b)
    llm = ChatOllama(
        base_url="http://localhost:11434",
        model="gemma3:1b"
    )
    print("✅ LLM(gemma3:1b) 초기화 완료")

    # 2. 도구(Tools) 준비

    # (1) QnA 에이전트 도구
    qna_agent_instance = QnAAgent()
    qna_tool = Tool(
        name="PDF_QnA",
        func=qna_agent_instance.answer,
        description="SPRi AI 산업동향 PDF 문서에 대한 질문에 답변할 때 사용합니다. 입력값은 질문 문자열입니다."
    )

    # (2) Tavily 검색 에이전트 도구
    tavily_agent_instance = TavilySearchAgent()
    search_tool = tavily_agent_instance.get_general_search_tool()

    # (3) 커스텀 계산 도구 (add_numbers, multiply_numbers)

    # 모든 도구를 리스트로 통합
    tools = [
        qna_tool,
        search_tool,
        add_numbers,
        multiply_numbers
    ]
    
    print(f"✅ 사용 가능한 도구: {[t.name for t in tools]}")

    # 3. 에이전트 초기화 (Manual ReAct Agent Construction)
    # 텍스트 기반의 ReAct 에이전트를 수동으로 구성합니다.

    # ReAct 프롬프트 템플릿 정의
    template = """Answer the following questions as best you can. You have access to the following tools:

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

    prompt = PromptTemplate.from_template(template)

    # 도구 설명 렌더링
    tool_names = ", ".join([t.name for t in tools])
    
    # 프롬프트에 도구 정보 주입
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=tool_names,
    )

    # LLM에 stop sequence 바인딩 (Observation 앞에서 멈추도록)
    llm_with_stop = llm.bind(stop=["\nObservation:"])

    # 에이전트 체인 구성 (RunnableSequence)
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | ReActSingleInputOutputParser()
    )

    # 에이전트 실행기 생성
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )
    print("✅ 에이전트 생성 완료 (Manual ReAct Agent)")

    # 4. 에이전트 실행 테스트
    print("\n========== 에이전트 테스트 시작 ==========")
    
    # 시나리오 1: PDF 문서 관련 질문
    query1 = "SPRi AI Brief에서 말하는 구글의 최신 동영상 생성 AI 모델 이름은 뭐야?"
    print(f"\n[질문 1] {query1}")
    agent_executor.invoke({"input": query1})

    # 시나리오 2: 웹 검색이 필요한 질문
    query2 = "현재 한국의 대통령은 누구야?"
    print(f"\n[질문 2] {query2}")
    agent_executor.invoke({"input": query2})

    # 시나리오 3: 계산이 필요한 질문
    query3 = "123 더하기 456은 몇이야? 그리고 그 결과에 2를 곱해줘."
    print(f"\n[질문 3] {query3}")
    agent_executor.invoke({"input": query3})

if __name__ == "__main__":
    main()
