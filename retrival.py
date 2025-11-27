from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



# 1. 로더 설정
# TextLoader를 사용하여 파일을 로드합니다.
loader = TextLoader("./data/appendix-keywords.txt", encoding="utf-8")

# 2. 문서를 로드합니다.
data = loader.load()
print(len(data))


# 3. 토크나이저 로드 (예외 처리 강화)
model_id = "google/gemma-3-1b-it" # 예시 모델 ID
try:
    # HuggingFace 토큰 인증이 필요할 수 있습니다. (huggingface-cli login 필요)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
except Exception as e:
    print(f"Gemma 토크나이저 로드 실패. 대체 토크나이저(gpt2)를 사용합니다. 오류: {e}")
    # 실습용으로 공개된 gpt2 토크나이저 사용 (토큰 수 계산 목적)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 4. 토큰 수를 계산하는 함수 정의
def count_tokens(text: str) -> int:
    # encode가 더 정확하고 빠를 수 있습니다.
    return len(tokenizer.encode(text))

# 5. Text Splitter 설정 (RecursiveCharacterTextSplitter 권장)
#text_splitter = RecursiveCharacterTextSplitter(
#    chunk_size=300,
#    chunk_overlap=0,
#    length_function=count_tokens, # 위에서 정의한 함수 사용
    # 아래 순서대로 분할을 시도합니다.
#    separators=["\n\n", "\n", " ", ""]
#)

# 문자 기반으로 텍스트를 분할하는 CharacterTextSplitter를 생성합니다. 청크 크기는 300이고 청크 간 중복은 없습니다.
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)


# WebBaseLoader는 페이지 전체를 하나의 Document로 가져오는 경우가 많으므로 split_documents 사용
texts = text_splitter.split_documents(data)

# 분할된 청크 확인
if texts:
    print(f"총 {len(texts)}개의 청크로 분할되었습니다.")
    #print(f"첫 번째 청크 예시:\n{texts[0].page_content[:200]}...")
else:
    print("분할된 텍스트가 없습니다. 크롤링 데이터를 확인해주세요.")
    exit()


# HuggingFaceE
embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sbert-nli',
    encode_kwargs={'normalize_embeddings':True},
)

# 6. 임베딩 및 벡터 저장소 (Chroma)
#embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# {'hnsw:space': 'cosine'} 벡터 공간에서 두 점(텍스트) 사이의 거리를 잴 때, '각도(Cosine)'를 기준으로 재겠다는 의미미
db = Chroma.from_documents(
    texts, 
    embeddings_model,
    collection_name='ppendix-keywords',
    persist_directory='./db/chromadb',
    collection_metadata={'hnsw:space': 'cosine'},
)


# 7. 검색 (Retrieval)
query = "임베딩(Embedding)은 무엇인가요?"

# 가장 유사도가 높은 문장을 하나만 추출
retriever = db.as_retriever(search_kwargs={'k': 1})

# 최신 LangChain 문법인 invoke 사용
docs = retriever.invoke(query)

for doc in docs:
    print(doc.page_content)
    print("=========================================================")