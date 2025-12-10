from langchain.tools import tool


# 데코레이터를 사용하여 함수를 도구로 변환합니다.
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@tool("곱셈연산")
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

# 도구 실행
add_numbers.invoke({"a": 3, "b": 4})

# 도구 실행
multiply_numbers.invoke({"a": 3, "b": 4})

print(f"도구 이름: {add_numbers.name}")   # add_numbers
print(f"도구 설명: {add_numbers.description}") # Add two numbers

# 사용자가 입력한 값을 도구에 전달
print(f"도구 이름: {multiply_numbers.name}")   # multiply_numbers

# LangChain이 생성한 args_schema를 확인해봅니다.
schema = add_numbers.args_schema # .args 속성을 통해 스키마에 접근 가능
print(schema)


from langchain_ollama import ChatOllama

tools = [add_numbers, multiply_numbers] #, python_tool]

llm = ChatOllama(
    base_url="http://localhost:11434",  # Ollama 서버 주소
    model="kimjk/llama3.2-korean",
    temperature=0.3,
)

llm_with_tools = llm.bind_tools(tools)
result_llm = llm_with_tools.invoke("1 + 3 = ?")
print(result_llm.content)