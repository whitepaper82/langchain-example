import os
import langchain
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# LangSmith 연동을 위한 traceable 핸들러를 가져옵니다.
from langsmith import traceable
# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from transformers import AutoTokenizer

from langchain_classic import hub  # langchain 1.0
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate

# API 키 정보 로드
load_dotenv()


## 토크나이저 huggingface에서 로드드
model_id = "google/gemma-3-12b-it"

try:
    # 토큰이 설정되어 있다고 가정하거나, 직접 입력 (보안상 환경변수 권장)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"'{model_id}' 토크나이저 로드 성공")
except Exception as e:
    print(f"⚠️ 토크나이저 로드 실패. 대체 모델(gpt2)을 사용합니다.\n오류: {e}")
    # 실패 시 프로그램이 멈추지 않도록 대체재 로드
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

##  토큰 수를 계산하는 함수 정의
# LangChain Text Splitter가 토큰 수를 셀 때 사용할 함수
def count_gemma_tokens(text: str) -> int:
    # return len(tokenizer.encode(text)) # encode는 토큰 ID 리스트를 반환
    return len(tokenizer.tokenize(text)) # tokenize는 토큰 문자열 리스트를 반환

## splitter 준비
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 청크당 목표 토큰 수
    chunk_overlap=50,   # 겹치는 토큰 수
    length_function=count_gemma_tokens, # 위에서 정의한 HuggingFace 토크나이저 함수 사용
    separators=["\n\n", "\n", ".", " ", ""], # 우선순위에 따라 쪼개 들어감
)

## LLM 모델 준비비
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="gemma3:1b"
)

# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader('./data/SPRi AI Brief_11월호_산업동향_1105_F.pdf')
data = loader.load()

# 단계 2: 문서 분할(Split Documents)
split_documents = text_splitter.split_documents(data)
print(f"문서 수: {len(data)}")
print(f"분할된 청크의수: {len(split_documents)}")

# 단계 3: 임베딩(Embedding) 생성
# embeddings = OllamaEmbeddings(model="nomic-embed-text")

# huggingface 임베딩 모델, 한국어 임베딩 모델 사용
embeddings = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sbert-nli',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)
#print(f"✅ Ollama 임베딩 모델: {embeddings.model} 연결 완료.")

# 단계 4: DB 생성(Create DB) 및 저장
vectorstore = Chroma.from_documents(
    split_documents, 
    embeddings,
    collection_name = 'SPRi',
    persist_directory = './db/chromadb',
    collection_metadata = {'hnsw:space': 'cosine'}, # l2 is the default
)

# 단계 5: 검색기(Retriever) 생성
retriever = vectorstore.as_retriever(search_kwargs={'k': 3}) 
# [디버깅] 실제로 무엇을 검색했는지 확인하는 코드
docs = retriever.invoke("구글의 최신 동영상 생성 AI 모델 이름은?")
print("=== 검색된 문서 내용 확인 ===")
for i, doc in enumerate(docs):
    print(f"[문서 {i+1}] {doc.page_content[:100]}...") # 앞부분 100자만 출력
print("===========================")

#print(retriever.invoke("구글의 최신 동영상 생성 AI 모델 이름은?"))

# 프롬프트 생성(Create Prompt)
#prompt = hub.pull("rlm/rag-prompt")
#print(prompt)
prompt = PromptTemplate.from_template(
    """당신은 질문에 답변하는 친절한 AI 어시스턴트입니다.
아래의 [Context]에 있는 내용만 사용하여 질문에 답하세요.
만약 [Context]에 정답이 있다면, 그것을 정답으로 간주하고 답변하세요.
답변은 반드시 '한국어'로 작성해야 합니다.

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#question = "구글의 최신 동영상 생성 AI 모델 이름은?"
#response = chain.invoke(question)
#print("\n\n응답 메시지 : ",response)

print("\n" + "="*50)
print("RAG 질문-응답 시스템이 준비되었습니다.")
print("종료하려면 'q', 'quit', 또는 '종료'를 입력하세요.")
print("="*50)

while True:
    question = input("\n질문을 입력하세요: ").strip()

    if question.lower() in ['q', 'quit', 'exit', '종료']:
        print("프로그램을 종료합니다.")
        break

    if not question:
        print("질문을 입력해주세요.")
        continue

    response = chain.invoke(question)
    print("\n응답 메시지:", response)