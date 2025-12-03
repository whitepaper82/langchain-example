from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv
load_dotenv()

tool = TavilySearchResults(max_results=1)
result = tool.invoke("antigravity는 무엇이고 최근 이슈는 무엇인가요?")
#response = client.search(query="antigravity는 무엇이고 최근 이슈는 무엇인가요?", search_depth="advanced", include_answer=True) 
print(result)


## topic
news_search_tool = TavilySearchResults(
    max_results=2,
    topic="news"
)
result = news_search_tool.invoke("antigravity는 무엇이고 최근 이슈는 무엇인가요?")
print("\n--------------------------------\n")
print(result)


### pythonREPLTool
from langchain.tools import PythonREPLTool

# 1. 도구 인스턴스 생성
python_tool = PythonREPLTool()

# 2. 코드 실행 (print() 사용 필수)
code_to_run = "import math; print(math.sqrt(144) + 5)"
result = python_tool.run(code_to_run)
print("\n--------------------------------\n")
print(result)
# result에는 '17.0\n'과 같은 실행 결과가 포함됩니다.

## 참고 사이트트
## https://github.com/MnMTech-hub/tutorials/blob/master/AI-Agents/Web-Agent.ipynb