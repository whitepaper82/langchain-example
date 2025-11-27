import pandas as pd

from langchain_community.document_loaders import PyMuPDFLoader
# (필요 시 DirectoryLoader 등 다른 loader 사용 가능)

from langchain_ollama import ChatOllama, OllamaEmbeddings

from ragas.testset import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context


# — 1. LLM & Embeddings 세팅 (Ollama 사용 예시) —  
llm = ChatOllama(
    base_url="http://localhost:11434",  # Ollama 서버 주소
    model="gemma3:1b",
    temperature=0.3,
)

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"  # embedding-support되는 모델 이름
)

# — 2. 문서 로드 —  
# 예: PDF 파일 하나를 로드  
loader = PyMuPDFLoader("data/SPRI_AI_Brief_2023년12월호_F.pdf")
docs = loader.load()

# 중요: 메타데이터에 'filename' (또는 비슷한 문서 식별자) 포함되어 있는지 확인
for doc in docs:
    if "filename" not in doc.metadata:
        # 예: source 필드를 filename으로 복사
        doc.metadata["filename"] = doc.metadata.get("source", "")

# — 3. RAGAS TestsetGenerator 초기화 —  
generator = TestsetGenerator.from_langchain(
    generator_llm=llm,
    critic_llm=llm,
    embeddings=embedding_model,
)

# — 4. testset 생성 —  
print("Ollama + RAGAS: testset 생성 시작...")

testset = generator.generate_with_langchain_docs(
    documents=docs,
    test_size=5,  # 생성할 질문 수
    distributions={
        simple: 0.5,        # 단순 질문
        reasoning: 0.25,    # 추론 질문
        multi_context: 0.25 # 다중 문맥 질문
    },
    with_debugging_logs=True,
    raise_exceptions=False, # 오류 발생 시 멈추지 않고 계속 진행
)

# — 5. 결과를 DataFrame으로 저장 및 출력 —  
df = testset.to_pandas()
df.to_csv("my_rag_testset_ollama.csv", index=False)

print("\n=== 생성된 데이터셋 (상위 3개) ===")
print(df[['question', 'ground_truth', 'context', 'evolution_type']].head(3))
