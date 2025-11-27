import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # 변경됨
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer
import torch

# 1. 타겟 URL 설정 (GeekNews)
url = "https://news.hada.io"

# 2. 로더 설정
# GeekNews의 구조상 topic_row 안의 텍스트들이 뭉쳐 보일 수 있어, separator를 명시하는 것이 좋습니다.
loader = WebBaseLoader(
    web_paths=(url,),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_="topic_row")
    ),
)

data = loader.load()
print(f"스크래핑 된 문서 개수: {len(data)}")

# 3. 토크나이저 로드 (예외 처리 강화)
model_id = "google/gemma-2-9b-it" # 예시 모델 ID
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
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=count_tokens, # 위에서 정의한 함수 사용
    # 아래 순서대로 분할을 시도합니다.
    separators=["\n\n", "\n", " ", ""]
)

# WebBaseLoader는 페이지 전체를 하나의 Document로 가져오는 경우가 많으므로 split_documents 사용
texts = text_splitter.split_documents(data)

# 분할된 청크 확인
if texts:
    print(f"총 {len(texts)}개의 청크로 분할되었습니다.")
    print(f"첫 번째 청크 예시:\n{texts[0].page_content[:200]}...")
else:
    print("분할된 텍스트가 없습니다. 크롤링 데이터를 확인해주세요.")
    exit()

# 6. 임베딩 및 벡터 저장소 (Chroma)
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# {'hnsw:space': 'cosine'} 벡터 공간에서 두 점(텍스트) 사이의 거리를 잴 때, '각도(Cosine)'를 기준으로 재겠다는 의미미
db = Chroma.from_documents(
    texts, 
    embeddings_model,
    collection_name='geeknews',
    persist_directory='./db/chromadb',
    collection_metadata={'hnsw:space': 'cosine'},
)


# 7. 검색 (Retrieval)
query = '최고의 디자이너들이 가진 습관들에 대해서 간략하게 알려주세요'

# 가장 유사도가 높은 문장을 하나만 추출
retriever = db.as_retriever(search_kwargs={'k': 1})

# 최신 LangChain 문법인 invoke 사용
docs = retriever.invoke(query)

print(f"\n검색된 문서 개수: {len(docs)}")
if docs:
    print("-" * 30)
    print(f"가장 유사한 문서 내용:\n{docs[0].page_content}")
    print("-" * 30)