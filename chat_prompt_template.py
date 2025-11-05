# 2-튜플 형태의 메시지 목록으로 프롬프트 생성 (type, content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from langchain_core.prompts import SystemMessagePromptTemplate,  HumanMessagePromptTemplate

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="gemma3:4b"
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "이 시스템은 천문학 질문에 답변할 수 있습니다."),
    ("user", "{user_input}"),
])

messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")
print("tuple 형태의 메시지 목록 : ",messages)

chain = chat_prompt | llm | StrOutputParser()

result = chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})
print("\ntuple 형태의 메시지 목록 처리 결과 : ",result)

chat_prompt_class = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("이 시스템은 천문학 질문에 답변할 수 있습니다."),
        HumanMessagePromptTemplate.from_template("{user_input}"),
    ]
)

messages_class = chat_prompt_class.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")
print("\nclass 형태의 메시지 목록 : ",messages_class)

chain_class = chat_prompt_class | llm | StrOutputParser()
result_class = chain_class.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})
print("\nclass 형태의 메시지 목록 처리 결과 : ",result_class) 

