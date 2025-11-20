from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# -----------------------------------------------------
# 1. Ollama 임베딩 객체 생성
# -----------------------------------------------------
# model 매개변수에 Ollama에 다운로드된 임베딩 모델 이름을 지정합니다.
# OllamaEmbeddings는 기본적으로 http://localhost:11434 에 연결을 시도합니다.
embeddings = OllamaEmbeddings(model="nomic-embed-text")
print(f"✅ Ollama 임베딩 모델: {embeddings.model} 연결 완료.")

# -----------------------------------------------------
# 2. 단일 텍스트 임베딩 생성 (벡터화)
# -----------------------------------------------------
text1 = "대규모 언어 모델은 인공지능 분야에서 중요한 역할을 합니다."

# embed_query 메서드를 사용하여 단일 텍스트를 벡터로 변환
vector1 = embeddings.embed_query(text1)

print("\n--- 텍스트 1 임베딩 결과 ---")
print(f"원본 텍스트: {text1}")
print(f"생성된 벡터의 차원(길이): {len(vector1)}")
# 벡터의 앞 5개 요소만 출력
print(f"벡터 (일부): {vector1[:5]}...")

# -----------------------------------------------------
# 3. 문서 목록 임베딩 생성 (RAG 활용)
# -----------------------------------------------------
# RAG 시스템에서 여러 청크(문서)를 벡터화할 때 사용
documents = [
    "태양계에서 가장 큰 행성은 목성입니다.",
    "사과는 빨간색이거나 녹색일 수 있습니다.",
    "Ollama는 로컬에서 LLM을 실행하기 위한 프레임워크입니다."
]

# embed_documents 메서드를 사용하여 여러 문서를 벡터 목록으로 변환
vector_list = embeddings.embed_documents(documents)

print("\n--- 문서 목록 임베딩 결과 ---")
print(f"총 문서 개수: {len(documents)}개")
print(f"총 생성된 벡터 개수: {len(vector_list)}개")
print(f"첫 번째 문서 벡터의 차원: {len(vector_list[0])}")

# -----------------------------------------------------
# 4. (선택 사항) 벡터 간 유사도 확인
# -----------------------------------------------------
# RAG의 기본 개념: 유사한 의미를 가진 텍스트는 벡터 공간에서 가까이 위치합니다.
from numpy.linalg import norm
import numpy as np

# 'ollama'와 관련된 문서의 벡터
vector_ollama = vector_list[2]
# '태양계'와 관련된 문서의 벡터
vector_solar = vector_list[0]

# 임베딩 벡터의 유사도를 측정하는 코사인 유사도(Cosine Similarity) 계산
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# (주의: 'nomic-embed-text' 벡터 크기가 크므로 numpy 변환 필요)
similarity = cosine_similarity(np.array(vector_ollama), np.array(vector_solar))

print("\n--- 벡터 유사도 확인 ---")
print(f"Ollama 문서와 태양계 문서 간의 코사인 유사도: {similarity:.4f}")
# 유사도가 0에 가까울수록 관련 없음. 1에 가까울수록 의미가 유사함.