# pip install beautifulsoup4
import bs4
from langchain_community.document_loaders import WebBaseLoader

# 1. 타겟 URL 설정 (GeekNews)
url = "https://news.hada.io"

# 2. 로더 설정
loader = WebBaseLoader(
    web_paths=(url,),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            # GeekNews에서 각 뉴스 아이템을 감싸고 있는 클래스명입니다.
            # 개발자 도구(F12)로 확인해보면 각 행이 'topic_row' div로 되어 있습니다.
            class_=("topic_row") 
        )
    ),
)

# 3. 데이터 로드
docs = loader.load()

# 4. 결과 확인
print(f"로드된 문서 개수: {len(docs)}")

if docs:
    print("\n--- [수집된 내용 일부 확인] ---")
    # 공백 정리 후 출력
    content = docs[0].page_content.strip()
    print(content[:500]) # 앞부분 500자만 출력
    print("...")