### pythonREPLTool
# pip install langchain-experimental
from langchain_experimental.tools import PythonREPLTool
from dotenv import load_dotenv
load_dotenv()

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