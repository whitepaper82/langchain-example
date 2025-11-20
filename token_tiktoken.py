from langchain_text_splitters import CharacterTextSplitter

# 1. 원본 텍스트 정의
# 파이썬의 멀티라인 문자열(""" """)을 사용하여 줄바꿈을 그대로 유지합니다.
text = """인공지능은 매우 중요합니다.
우리의 삶을 바꿀 것입니다.

하지만 위험성도 있습니다."""


text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=25,
    chunk_overlap=0,
    encoding_name='cl100k_base'
)

chunks = text_splitter.split_text(text)
print(f"\n총 청크 개수: {len(chunks)}개")
print("-" * 30)
for i, chunk in enumerate(chunks):
    print(f"[Chunk {i+1}] 길이: {len(chunk)}")
    # 줄바꿈(\n)이 출력에서 헷갈리지 않도록 시각적으로 표현 (.replace)
    print(f"내용: {chunk.replace(chr(10), '(줄바꿈)')}")
    print("-" * 30)