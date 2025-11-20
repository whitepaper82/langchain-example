from langchain_community.embeddings import HuggingFaceEmbeddings
from numpy.linalg import norm
import numpy as np

# -----------------------------------------------------
# 1. HuggingFaceEmbeddings 객체 생성
# -----------------------------------------------------
# 사용할 임베딩 모델 이름을 지정합니다. (Hugging Face Hub에 있는 모델)
model_name = "all-MiniLM-L6-v2"

# HuggingFaceEmbeddings 객체 생성
# device='cpu'는 GPU가 없을 경우 명시적으로 CPU 사용을 지정합니다.
# (GPU가 있다면 device='cuda'를 사용할 수 있습니다.)
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'}
)

print(f"✅ Hugging Face 임베딩 모델: {model_name} 로드 완료.")

# -----------------------------------------------------
# 2. 문서 목록 임베딩 생성
# -----------------------------------------------------
# RAG 시스템에서 벡터화할 텍스트 문서 목록
documents = [
    "태양계에서 가장 큰 행성은 목성입니다.",
    "애플은 새로운 아이폰 모델을 공개했습니다.",
    "가장 빠른 LLM 프레임워크는 무엇일까요?",
    "태양계 행성의 크기에 대한 정보가 필요해.",
]

# embed_documents 메서드를 사용하여 여러 문서를 벡터 목록으로 변환
vector_list = embeddings.embed_documents(documents)

print("\n--- 임베딩 결과 요약 ---")
print(f"총 문서 개수: {len(documents)}개")
print(f"총 생성된 벡터 개수: {len(vector_list)}개")
# 'all-MiniLM-L6-v2' 모델은 384차원 벡터를 생성합니다.
print(f"벡터의 차원: {len(vector_list[0])}") 

# -----------------------------------------------------
# 3. 벡터 유사도 검색 및 확인
# -----------------------------------------------------
query = "새로운 IT 기기에 대한 소식을 알고 싶어."

# 검색어(Query)를 벡터로 변환
query_vector = embeddings.embed_query(query)

# 코사인 유사도 계산 함수
def cosine_similarity(a, b):
    # 벡터를 numpy 배열로 변환
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (norm(a) * norm(b))

print("\n--- 검색 결과 유사도 ---")
for i, doc_vector in enumerate(vector_list):
    similarity = cosine_similarity(query_vector, doc_vector)
    
    # 유사도(Similarity)와 원본 문서(Source Document)를 출력
    print(f"유사도: {similarity:.4f} | 문서: \"{documents[i]}\"")