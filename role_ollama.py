import ollama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# 1. LangChain ChatOllama 모델 초기화
# 원본 ollama.chat의 options을 모델 초기화 시점에 지정합니다.
# (temperature, num_ctx, num_predict 등)
llm = ChatOllama(
    model='gemma3:4b',
    temperature=0.2,
    num_ctx=1024,
    num_predict=1024 #200
)

# --- 원본 프롬프트 정의 ---
prompt_basic = "인공지능의 미래에 대해 설명해줘."

prompt_role = """
너는 세계 최고의 SF 소설가야.
독자들이 흥미진진하게 읽을 수 있도록, 인공지능의 미래에 대해 한 편의 짧은 소설처럼 묘사해줘.
"""

print("--- [기본 프롬프트 결과 (LangChain)] ---")

# 2. LangChain 메시지 형식으로 변환 후 'invoke' 호출
# 'ollama.chat' 대신 'llm.invoke'를 사용합니다.
# 메시지 형식을 딕셔너리가 아닌 HumanMessage 객체로 전달합니다.
messages_basic = [
    HumanMessage(content=prompt_basic)
]
response_basic = llm.invoke(messages_basic)

# 3. 결과 출력
# 결과는 딕셔너리가 아닌 AIMessage 객체이며, .content로 접근합니다.
print(response_basic.content)


print("\n\n--- [역할 부여 프롬프트 결과 (LangChain - 원본 방식)] ---")

# 원본 코드는 역할과 질문을 모두 HumanMessage에 담았습니다.
# LangChain에서도 동일하게 작동합니다.
messages_role_original = [
    HumanMessage(content=prompt_role)
]
response_role_original = llm.invoke(messages_role_original)

print(response_role_original.content)


print("\n\n--- [역할 부여 프롬프트 결과 (LangChain - 권장 방식)] ---")

# LangChain의 장점은 '역할'을 SystemMessage로 명확히 분리하는 것입니다.
# 이렇게 하면 모델이 역할을 더 잘 이해하고 일관된 답변을 생성합니다.
messages_role_idiomatic = [
    SystemMessage(content="너는 세계 최고의 SF 소설가야."),
    HumanMessage(content="독자들이 흥미진진하게 읽을 수 있도록, 인공지능의 미래에 대해 한 편의 짧은 소설처럼 묘사해줘.")
]
response_role_idiomatic = llm.invoke(messages_role_idiomatic)

print(response_role_idiomatic.content)


print("\n\n--- [역할 부여 프롬프트 결과 (딕셔너리 리스트 방식)] ---")

# 2. 딕셔너리 리스트 형식으로 프롬프트 정의
messages_role_dict_format = [
    {'role': 'system', 'content': "너는 세계 최고의 SF 소설가야."},
    {'role': 'user', 'content': "독자들이 흥미진진하게 읽을 수 있도록, 인공지능의 미래에 대해 한 편의 짧은 소설처럼 묘사해줘."}
]

# 3. llm.invoke()에 딕셔너리 리스트를 그대로 전달
response_role_dict = llm.invoke(messages_role_dict_format)
print(response_role_dict.content)