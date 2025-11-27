import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# 1. 데이터 준비 (Data Preparation)
# ---------------------------------------------------------
# 상황: 'Apple'이라는 단어는 '과일'과 'IT 기업' 두 가지 의미가 있습니다.
# 문장 목록 (Documents)
documents = [
    "Apple is a tasty red fruit",           # 문장 0: 과일 관련 (사과는 맛있는 빨간 과일이다)
    "Apples and bananas are healthy",       # 문장 1: 과일 관련 (사과와 바나나는 건강에 좋다)
    "Apple announced the new iPhone",       # 문장 2: 기업 관련 (애플은 새 아이폰을 발표했다)
    "Steve Jobs founded Apple company",     # 문장 3: 기업 관련 (스티브 잡스는 애플 회사를 창립했다)
    "The Galaxy phone is from Samsung"      # 문장 4: 기업 관련 (갤럭시 폰은 삼성 제품이다) - 경쟁사(다양성 테스트용)
]

# 사용자 질문 (Query)
# 질문이 모호합니다. 'Apple'에 대해 묻고 있지만, 과일인지 기업인지 명확하지 않습니다.
query = ["Tell me about Apple"] 

# 전체 텍스트 뭉치 (질문 포함)
all_text = documents + query

# ---------------------------------------------------------
# 2. 벡터화 (Vectorization: Text -> Numbers)
# ---------------------------------------------------------
# TF-IDF 방식을 사용하여 문장을 벡터로 변환합니다.
# (단어의 빈도와 희소성을 고려하여 중요도를 숫자로 바꿈)
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(all_text)

# 문서 벡터들과 질문 벡터를 분리
doc_vectors = tfidf_matrix[:-1].toarray()
query_vector = tfidf_matrix[-1].toarray()

# ---------------------------------------------------------
# 3. 코사인 유사성 계산 (Cosine Similarity)
# ---------------------------------------------------------
# 질문 벡터와 각 문서 벡터 사이의 유사도를 계산합니다.
# 결과값은 -1 ~ 1 사이이며, 1에 가까울수록 질문과 유사합니다.
cosine_similarities = cosine_similarity(query_vector, doc_vectors).flatten()

print("=== [1] 코사인 유사성 순위 (단순 유사도) ===")
# 유사도 순으로 정렬하여 출력
sorted_indices = cosine_similarities.argsort()[::-1]
for idx in sorted_indices:
    print(f"문서 {idx} (유사도: {cosine_similarities[idx]:.4f}): {documents[idx]}")
print("\n")

# ---------------------------------------------------------
# 4. MMR (Max Marginal Relevance) 알고리즘 구현
# ---------------------------------------------------------
# 목적: 질문과 관련이 있으면서도(Relevance), 이미 선택된 문서와는 다른(Diversity) 문서를 뽑는다.
# 공식: MMR Score = lambda * (질문과의 유사도) - (1 - lambda) * (선택된 문서들과의 유사도 중 최대값)

def mmr_selection(doc_vectors, query_vector, docs, lambda_param=0.5, top_k=3):
    selected_indices = []
    candidate_indices = list(range(len(docs)))
    
    print(f"=== [2] MMR 결과 (다양성 고려, Lambda={lambda_param}) ===")

    for step in range(top_k):
        best_score = -np.inf
        best_doc_idx = -1
        
        for doc_idx in candidate_indices:
            # 1. 질문과의 유사도 (Relevance)
            relevance = cosine_similarity([doc_vectors[doc_idx]], query_vector)[0][0]
            
            # 2. 이미 선택된 문서들과의 유사도 (Diversity Penalty)
            if not selected_indices:
                diversity_penalty = 0
            else:
                # 선택된 문서들과 현재 문서 사이의 유사도 중 가장 큰 것(가장 겹치는 것)을 찾음
                sim_to_selected = cosine_similarity([doc_vectors[doc_idx]], doc_vectors[selected_indices])
                diversity_penalty = np.max(sim_to_selected)
            
            # 3. MMR 점수 계산
            mmr_score = (lambda_param * relevance) - ((1 - lambda_param) * diversity_penalty)
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_doc_idx = doc_idx
        
        # 선택된 문서를 결과에 추가하고 후보에서 제거
        selected_indices.append(best_doc_idx)
        candidate_indices.remove(best_doc_idx)
        print(f"순위 {step+1}: 문서 {best_doc_idx} (MMR 점수: {best_score:.4f}) -> {docs[best_doc_idx]}")
    
    return selected_indices

# MMR 실행
mmr_selection(doc_vectors, query_vector, documents, lambda_param=0.5, top_k=3)


# ---------------------------------------------------------
# 5. 시각화 (Visualization) - 2D Plot
# ---------------------------------------------------------
# 고차원 벡터를 2차원 평면에 그리기 위해 PCA(차원 축소) 사용
pca = PCA(n_components=2)
coords = pca.fit_transform(tfidf_matrix.toarray())

doc_coords = coords[:-1]
query_coord = coords[-1]

plt.figure(figsize=(10, 8))
plt.title("Vector Space Visualization (Query vs Documents)", fontsize=15)

# 원점 (0,0) 설정
origin = [0], [0]

# 1. 문서 벡터 그리기 (파란색 화살표)
for i, (x, y) in enumerate(doc_coords):
    plt.quiver(*origin, x, y, color='skyblue', scale=1, scale_units='xy', angles='xy', width=0.005)
    plt.text(x, y, f"Doc {i}\n({documents[i][:15]}...)", fontsize=9, color='blue')

# 2. 질문 벡터 그리기 (빨간색 화살표)
plt.quiver(*origin, query_coord[0], query_coord[1], color='red', scale=1, scale_units='xy', angles='xy', width=0.008, label='Query')
plt.text(query_coord[0], query_coord[1], " QUERY (Apple)", fontsize=12, color='red', fontweight='bold')

# 그래프 설정
plt.grid(True, linestyle='--', alpha=0.5)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim(-0.8, 0.8)
plt.ylim(-0.8, 0.8)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()

# 이미지 저장 (화면에 보여주기 위함)
plt.savefig('vector_similarity_plot.png')
print("\n=== [3] 시각화 완료 ===")
print("'vector_similarity_plot.png' 파일이 생성되었습니다. (그래프 확인)")