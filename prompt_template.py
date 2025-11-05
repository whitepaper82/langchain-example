from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
# 환경 변수 로드
load_dotenv()

# 1. 기본 PromptTemplate 정의
# 'table_number'와 'guest_name' 변수를 사용
template_text = "테이블 번호 {table_number}번, 주문자 {guest_name}입니다."

# PromptTemplate 인스턴스 생성
prompt_template = PromptTemplate.from_template(template_text)

# 2. '+' 연산자를 사용하여 템플릿 결합
# 기존 prompt_template에 새로운 템플릿('main_dish' 변수)과 
# 문자열('drink' 변수)을 결합합니다.
combined_prompt = (
    prompt_template
    + PromptTemplate.from_template("\n\n메인 요리는 {main_dish}(으)로 주세요.")
    + "\n음료는 {drink}로 부탁드립니다."
)


# 3. LLM 체인 구성 및 실행
# 이 주문서를 요약해서 주방에 전달하는 LLM을 가정합니다.
#llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="gemma3:1b"
)

prompt_for_llm = (
    combined_prompt
    + "\n\n---"
    + "\n위 주문 내역을 주방에 전달할 수 있도록 한 줄로 요약해주세요."
)
print(prompt_for_llm + "\n\n---")

chain = prompt_for_llm | llm | StrOutputParser()

# 체인 실행: 4개의 변수를 모두 제공
result = chain.invoke({
    "table_number": 7,
    "guest_name": "이영희",
    "main_dish": "까르보나라 파스타",
    "drink": "아이스 아메리카노"
})

print(result)