from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.example_selectors import NGramOverlapExampleSelector

# 1. 예제 데이터
examples = [
    {"sentence": "Spot can run.", "translation": "Spot puede correr."},
    {"sentence": "My dog barks.", "translation": "Mi perro ladra."},
]

# 2. 개별 예제를 포맷팅할 템플릿
example_prompt = PromptTemplate(
    input_variables=["sentence", "translation"],
    template="Input: {sentence}\nOutput: {translation}"
)

# 3. N-gram Overlap Selector
example_selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    threshold=-1.0   # 점수 순 정렬 (기본값)
)

# 4. 동적 FewShot PromptTemplate 생성
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Translate the following English sentences into Spanish.",
    suffix="Input: {sentence}\nOutput:",
    input_variables=["sentence"],
    example_separator="\n\n"
)

# 5. 프롬프트 생성 테스트
print(dynamic_prompt.format(sentence="Spot can run fast."))