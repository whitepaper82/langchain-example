import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List

class TavilySearchAgent:
    """
    TavilySearchResults 도구를 사용하여 검색을 수행하는 클래스입니다.
    """
    
    def __init__(self, max_results: int = 5):
        """
        클래스를 초기화하고 환경 변수를 로드합니다.

        :param max_results: 기본 검색에서 가져올 최대 결과 수.
        """
        # .env 파일 로드 (TAVILY_API_KEY 설정 필요)
        load_dotenv()
        
        # TAVILY_API_KEY가 로드되었는지 확인
        if not os.getenv("TAVILY_API_KEY"):
            raise ValueError("TAVILY_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

        # 일반 검색 도구 초기화
        self._general_search_tool = TavilySearchResults(max_results=max_results)
        
        # 뉴스 주제 검색 도구 초기화 (최대 결과 2개, 주제 'news' 고정)
        self._news_search_tool = TavilySearchResults(
            max_results=2,
            topic="news"
        )
        print(f"✅ TavilySearchAgent 초기화 완료. 기본 검색 결과 수: {max_results}")

    def get_general_search_tool(self):
        """웹 일반 검색 도구(TavilySearchResults) 인스턴스를 반환합니다."""
        return self._general_search_tool

    def get_news_search_tool(self):
        """웹웹 뉴스 검색 도구(TavilySearchResults) 인스턴스를 반환합니다."""
        return self._news_search_tool

    def general_search(self, query: str) -> List[str]:
        """
        일반 검색 도구를 사용하여 쿼리를 실행합니다.

        :param query: 검색할 내용.
        :return: 검색 결과 문자열 리스트.
        """
        print(f"\n--- [일반 검색] 쿼리: '{query[:50]}...' ---")
        result = self._general_search_tool.invoke(query)
        return result

    def news_search(self, query: str) -> List[str]:
        """
        'news' 주제 검색 도구를 사용하여 쿼리를 실행합니다. (max_results=2)

        :param query: 검색할 내용.
        :return: 검색 결과 문자열 리스트.
        """
        print(f"\n--- [뉴스 검색] 쿼리: '{query[:50]}...' ---")
        result = self._news_search_tool.invoke(query)
        return result

# --- 사용 예시 ---
if __name__ == "__main__":
    # 1. 클래스 인스턴스 생성 (기본 검색 결과 수를 1로 설정)
    try:
        agent = TavilySearchAgent(max_results=1)

        query_string = "antigravity는 무엇이고 최근 이슈는 무엇인가요?"

        # 2. 일반 검색 실행 (max_results=1)
        general_result = agent.general_search(query_string)
        print("\n[일반 검색 결과 (max_results=1)]")
        for item in general_result:
            print(item)

        # 3. 뉴스 주제 검색 실행 (max_results=2, topic='news')
        news_result = agent.news_search(query_string)
        print("\n--------------------------------")
        print("\n[뉴스 검색 결과 (max_results=2, topic='news')]")
        for item in news_result:
            print(item)

    except ValueError as e:
        print(f"\n❌ 에이전트 초기화 실패: {e}")