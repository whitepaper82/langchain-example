from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

# 2. Ollama 채팅 모델 초기화
# model 파라미터에 로컬에 설치된 모델 이름을 지정합니다.
# 변경 예시: IP 주소 192.168.1.10의 12345 포트를 사용하는 경우
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="gemma3:1b"
)

# 3. 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

# 4. LangChain Expression Language (LCEL)을 사용해 체인 구성
chain = prompt | llm

# 5. 체인 실행
response = chain.invoke({"input": "대한민국의 수도는 어디인가요?"})

# content 속성으로 결과 확인
print(response)