from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_classic import hub  # langchain 1.0
from langchain_community.vectorstores import Chroma

import pandas as pd
from datasets import Dataset

df = pd.read_csv("data/ragas_dataset.csv")
test_dataset = Dataset.from_pandas(df)

import ast


# contexts 컬럼의 문자열을 리스트로 변환
def convert_to_list(example):
    contexts = ast.literal_eval(example["contexts"])
    return {"contexts": contexts}


test_dataset = test_dataset.map(convert_to_list)

### 여기까지 생성한 문서 로드


# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader("data/SPRI_AI_Brief_2023년12월호_F.pdf")
docs = loader.load()

# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

# 단계 3: 임베딩(Embedding) 생성
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"  # embedding-support되는 모델 이름
)

# 단계 4: DB 생성(Create DB) 및 저장
# 벡터스토어를 생성합니다.
vectorstore = Chroma.from_documents(
    documents=split_documents,
    embedding=embedding_model,
    collection_name='ragas_collection',
    persist_directory='./db/chromadb_ragas',
    collection_metadata={'hnsw:space': 'cosine'},
)

# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# 단계 6: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """당신은 질문에 답변하는 친절한 AI 어시스턴트입니다.
아래의 [Context]에 있는 내용만 사용하여 질문에 답하세요.
만약 [Context]에 정답이 있다면, 그것을 정답으로 간주하고 답변하세요.
답변은 반드시 '한국어'로 작성해야 합니다.

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)


# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.
llm = ChatOllama(
    base_url="http://localhost:11434",  # Ollama 서버 주소
    model="gemma3:4b",
    temperature=0.3,
)

# 단계 8: 체인(Chain) 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


#배치 데이터셋을 생성합니다. 배치 데이터셋은 다량의 질문을 한 번에 처리할 때 용이합니다.
batch_dataset = [question for question in test_dataset["question"]] #질문(question)' 컬럼만 뽑아서 파이썬 리스트(List)
answer = chain.batch(batch_dataset)

# LLM 이 생성한 답변을 'answer' 컬럼에 저장합니다.
# 'answer' 컬럼 덮어쓰기 또는 추가
if "answer" in test_dataset.column_names:
    test_dataset = test_dataset.remove_columns(["answer"]).add_column("answer", answer)
else:
    test_dataset = test_dataset.add_column("answer", answer)


### ragas 평가
#이 지표는 생성된 답변이 주어진 컨텍스트에 얼마나 충실한지를 평가하는 데 유용하며, 특히 질문-답변 시스템의 정확성과 신뢰성을 측정하는 데 중요합니다.
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset=test_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)

# output : {'context_precision': 0.8000, 'faithfulness': 0.6689, 'answer_relevancy': 0.7836, 'context_recall': 0.7667}

result_df = result.to_pandas()
print(result_df.head())
print(result_df.loc[:, "context_precision":"context_recall"])