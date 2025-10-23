# LangChain 예제 프로젝트

이 프로젝트는 LangChain을 사용한 다양한 예제들을 포함합니다.

## 파일 구조

- `openai_tool.py` - OpenAI Function Calling을 사용한 도구 선택 예제
- `openai_smith.py` - LangSmith 추적 기능이 포함된 예제
- `ollama_default.py` - Ollama 로컬 모델 사용 예제
- `requirements.txt` - 필요한 패키지 목록

## 설치 방법

1. 가상환경 생성 (권장):
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. 패키지 설치:
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정:
`.env` 파일을 생성하고 다음 내용을 추가하세요:
```
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
SERPAPI_API_KEY=your_serpapi_key
```

## 실행 방법

```bash
python openai_tool.py
```

## 주요 기능

- OpenAI Function Calling을 통한 도구 선택
- LangSmith를 통한 실행 추적
- 웹 검색 및 수학 계산 도구
- Ollama 로컬 모델 지원
