from transformers import AutoTokenizer
from langchain_text_splitters import CharacterTextSplitter

# 1. Gemma 3 토크나이저 불러오기
# Gemma 모델의 토크나이저와 동일한 것을 사용합니다.
# 실제 모델 이름(예: 'google/gemma-2-9b')을 사용해야 정확합니다.
try:
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
except Exception as e:
    # 예외 발생 시, 토크나이저 파일이 로컬에 없거나 네트워크 문제일 수 있음
    print(f"토크나이저 로드 실패. 'google/gemma-2-9b'가 최신 Gemma 3 모델의 예시입니다. 오류: {e}")
    # 대안으로 SentencePiece 기반의 다른 토크나이저를 사용할 수도 있지만, 정확성은 떨어짐
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")


# 2. 토큰 수를 계산하는 함수 정의
# LangChain Text Splitter가 토큰 수를 셀 때 사용할 함수
def count_gemma_tokens(text: str) -> int:
    # return len(tokenizer.encode(text)) # encode는 토큰 ID 리스트를 반환
    return len(tokenizer.tokenize(text)) # tokenize는 토큰 문자열 리스트를 반환


# 3. LangChain CharacterTextSplitter 설정
# 'separator'를 공백이나 개행 문자로 설정하여 텍스트를 나눌 기본 단위를 지정합니다.
text_splitter = CharacterTextSplitter(
    separator="\n\n", # 텍스트를 나눌 기본 구분자 (문단 기준)
    chunk_size=500, # 청크당 최대 토큰 수 (예시: 500 토큰)
    chunk_overlap=100, # 청크 간 겹치는 토큰 수
    length_function=count_gemma_tokens, # 토큰 수를 세는 함수 지정 (가장 중요)
)

# 4. 텍스트 분할 실행
long_text = "여기에 매우 긴 문서를 넣습니다. 예를 들어, 한 페이지가 넘는 긴 보고서나 책의 장을 넣어 보세요. 이 텍스트는 토크나이저를 통해 토큰으로 변환되며, 설정된 500 토큰 크기에 맞게 여러 조각으로 나누어질 것입니다. 토큰 기반 분할은 LLM의 입력 제한을 정확히 맞출 수 있게 해줍니다. 이 과정은 RAG 시스템에서 데이터를 벡터 DB에 저장하기 전에 필수적입니다. " * 10

# 분할된 청크 목록
chunks = text_splitter.create_documents([long_text])

# 결과 확인
print(f"원본 텍스트 길이: {len(long_text)} 문자")
print(f"생성된 청크 수: {len(chunks)}개")
print(f"첫 번째 청크의 내용:\n--- \n{chunks[0].page_content[:200]}...\n---")
# 첫 번째 청크의 실제 토큰 수 확인 (optional)
print(f"첫 번째 청크의 토큰 수: {count_gemma_tokens(chunks[0].page_content)}")

print(f"토큰큰 문자열: {tokenizer.tokenize(chunks[0].page_content)}")
