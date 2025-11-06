from langchain_core.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# --- 1. '예제(Examples)' 데이터 정의 ---
# 모델에게 학습시킬 "질문"과 "답변"의 쌍(패턴)입니다.
examples = [
    {
        "input": "귀사의 무궁한 발전을 기원합니다.",
        "output": "항상 응원할게요! 대박나세요."
    },
    {
        "input": "금일 중으로 회신 부탁드립니다.",
        "output": "오늘 안에 답장 줘."
    },
    {
        "input": "해당 사안에 대해 검토해 보았습니다.",
        "output": "그거 생각해 봤는데,"
    }
]

# --- 2. '예제 형식(Format)' 정의 ---
# 위 'examples'의 각 항목을 어떤 "일관된 형식"으로 보여줄지 정의합니다.
example_prompt = PromptTemplate(
    input_variables=["input", "output"], # examples의 key와 일치
    template="딱딱한 말: {input}\n친근한 말: {output}\n---" # 예시 하나하나의 형식
)

# --- 3. 'FewShotPromptTemplate' 조립 ---
# 인용하신 글의 "예제들을 결합하고 새로운 입력을 추가"하는 부분입니다.
few_shot_prompt = FewShotPromptTemplate(
    # ① "예제들을 결합"하는 부분
    examples=examples,                # 사용할 예시 데이터
    example_prompt=example_prompt,    # 예시를 포맷할 템플릿
    
    # 예시 목록이 시작되기 전, LLM에게 주는 전체 지시사항
    prefix="다음은 딱딱한 문어체 문장을 친근한 구어체로 바꾸는 예시입니다.",
    
    # ② "새로운 입력을 추가"하는 부분 (모델이 실제로 풀어야 할 문제)
    suffix="딱딱한 말: {user_input}\n친근한 말:", # 모델이 '친근한 말:' 뒤를 채우도록 유도
    
    input_variables=["user_input"]    # 사용자가 최종적으로 입력할 변수
)

# 4. LLM 체인 구성 및 실행

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="gemma3:4b"
)
chain = few_shot_prompt | llm | StrOutputParser()

# --- (참고) 최종 프롬프트가 어떻게 생겼는지 확인하기 ---
# final_prompt = few_shot_prompt.format(user_input="회의에 참석해 주셔서 감사합니다.")
# print(final_prompt)
#
# [출력 예시]
# 다음은 딱딱한 문어체 문장을 친근한 구어체로 바꾸는 예시입니다.
#
# 딱딱한 말: 귀사의 무궁한 발전을 기원합니다.
# 친근한 말: 항상 응원할게요! 대박나세요.
# ---
# 딱딱한 말: 금일 중으로 회신 부탁드립니다.
# 친근한 말: 오늘 안에 답장 줘.
# ---
# 딱딱한 말: 해당 사안에 대해 검토해 보았습니다.
# 친근한 말: 그거 생각해 봤는데,
# ---
# 딱딱한 말: 회의에 참석해 주셔서 감사합니다.
# 친근한 말:
#
# (LLM은 마지막 '친근한 말:' 뒤에 '와주셔서 고마워요!' 같은 답변을 생성합니다)
# ----------------------------------------------------


# --- 체인 실행 ---
#new_input = "회의에 참석해 주셔서 감사합니다."
new_input = "귀하의 입사를 진심으로 축하드립니다."
result = chain.invoke({"user_input": new_input})

print(f"입력: {new_input}")
print(f"결과: {result}")