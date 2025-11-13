from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# pip install langchain-ollama chromadb
# ollama pull nomic-embed-text

from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_ollama import OllamaEmbeddings  # OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import FewShotPromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

# 1. 예제 데이터 풀 정의
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "bad", "output": "good"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "hot", "output": "cold"},
]

# 2. 개별 예제를 포맷팅할 템플릿 정의
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# 1. 임베딩 모델 설정 (Ollama에서 지원하는 임베딩 모델 지정 필요)
# 주의: 'llama3.1' 같은 채팅 모델보다는 임베딩 전용 모델 사용을 권장합니다.
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"  # 사용할 Ollama 모델 이름
)

# 2. ExampleSelector 인스턴스화
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,                 # 사용할 예제 풀
    embeddings,               # Ollama 임베딩 모델 객체
    Chroma,                   # 벡터를 저장하고 검색할 VectorStore 클래스
    k=1                       # 입력과 가장 유사한 1개의 예제를 선택 
)

# 4. FewShotPromptTemplate 구성 (dynamic_prompt)
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,    # 정적 'examples' 대신 'example_selector' 사용
    example_prompt=example_prompt,
    prefix="모든 입력에 대해 반의어를 제공하세요.",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],        # 사용자로부터 'adjective' 입력을 받음
    example_separator="\n\n"
)
print(dynamic_prompt.format(adjective="windy"))  # windy


# 2. Ollama 채팅 모델 초기화
# model 파라미터에 로컬에 설치된 모델 이름을 지정합니다.
# 변경 예시: IP 주소 192.168.1.10의 12345 포트를 사용하는 경우
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="gemma3:1b"
)

# 5. 체인 실행
response = llm.invoke(dynamic_prompt.format(adjective="windy"))

# content 속성으로 결과 확인
print(response.content)