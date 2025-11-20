from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter

# 1. 원본 텍스트 정의
# 파이썬의 멀티라인 문자열(""" """)을 사용하여 줄바꿈을 그대로 유지합니다.
text = """인공지능은 매우 중요합니다.
우리의 삶을 바꿀 것입니다.

하지만 위험성도 있습니다."""

# 2. 분할기(Splitter) 설정
# chunk_size=25: 이보다 길면 자르기 시작합니다.
# chunk_overlap=0: 겹치는 구간 없이 깔끔하게 자릅니다.
recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=25,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""] # 기본값과 동일 (명시적으로 표시)
)

# 3. 텍스트 분할 실행
chunks = recursive_text_splitter.split_text(text)

# 4. 결과 출력
print(f"재귀분할 - 총 청크 개수: {len(chunks)}개")
print("-" * 30)
for i, chunk in enumerate(chunks):
    print(f"[Chunk {i+1}] 길이: {len(chunk)}")
    print(f"내용: {chunk}")
    print("-" * 30)

# 단순 분할기(CharacterTextSplitter) 설정
# separator="": 띄어쓰기나 줄바꿈 고려 없이 25자가 되면 무조건 자릅니다.
text_splitter = CharacterTextSplitter(
    chunk_size=25,
    chunk_overlap=0,
    separator=""  # 재귀적 분할이 없으므로 글자 단위로 자르도록 설정
)

# 3. 분할 실행
chunks = text_splitter.split_text(text)

# 4. 결과 출력
print(f"\n단순 분할 - 총 청크 개수: {len(chunks)}개")
print("-" * 30)
for i, chunk in enumerate(chunks):
    print(f"[Chunk {i+1}] 길이: {len(chunk)}")
    # 줄바꿈(\n)이 출력에서 헷갈리지 않도록 시각적으로 표현 (.replace)
    print(f"내용: {chunk.replace(chr(10), '(줄바꿈)')}")
    print("-" * 30)