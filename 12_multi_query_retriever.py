
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_classic.retrievers import MultiQueryRetriever


import logging

logging.basicConfig()
logging.getLogger('langchain_classic.retrievers').setLevel(logging.INFO)


# 모델 및 임베딩 준비
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="gemma3:4b", 
    temperature=0 
)

embeddings = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sbert-nli',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)

# 데이터 로드 및 저장 (기존 DB가 있다면 생략 가능)
# 실습을 위해 문서를 로드하고 DB에 넣는 과정을 포함했습니다.
loader = PyMuPDFLoader('./data/SPRi AI Brief_11월호_산업동향_1105_F.pdf')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    collection_name='SPRi_MultiQuery', # 컬렉션 이름 구분
    persist_directory='./db/chromadb_mq',
)

# =================================================================
# [핵심] 4. MultiQueryRetriever 설정
# =================================================================
# 일반 검색기(Base Retriever)를 먼저 만듭니다.
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM을 이용해 질문을 확장하는 MultiQueryRetriever로 감쌉니다.
query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""당신은 AI 언어 모델 어시스턴트입니다. 반드시 '한국어'로 작성하세요.
사용자의 질문을 보고, 벡터 데이터베이스에서 관련 문서를 찾기 위한 3가지 다른 버전의 검색 쿼리를 생성하세요.
다양한 관점에서 정보를 찾을 수 있도록 질문을 변형하세요.
각 질문은 줄바꿈으로 구분하고, 번호나 다른 말은 붙이지 마세요. 오직 질문만 출력하세요.

원본 질문: {question}"""
)

# from_llm의 내부 파라미터로 prompt를 전달
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    prompt=query_prompt,   # 커스텀 프롬프트 적용
    include_original=True, # 원본 질문도 검색에 포함
)

query = "구글의 최신 동영상 생성 AI 이름?"
# =================================================================
# 멀티 검색기 설정
# =================================================================
print("\n[멀티 검색 실행]")
retriver_docs = multi_retriever.invoke(query)

print(f"\n검색기 문서 개수: {len(retriver_docs)}")
for i, doc in enumerate(retriver_docs):
    print(f"\n[문서 {i+1}]")
    print(doc.page_content) 


# 5. RAG 체인 구성
prompt = PromptTemplate.from_template(
    """당신은 친절한 AI 어시스턴트입니다. 아래 문맥을 보고 질문에 답하세요.
답변은 반드시 '한국어'로 작성하세요.

#Context: 
{context} 

#Question: 
{question} 

#Answer:"""
)

chain = (
    {"context": multi_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. 실행 및 결과 확인
"""
question = query
print(f"\n사용자 질문: {question}")
print("-" * 50)

# invoke 시 로그에 'Generated queries'가 출력됩니다.
response = chain.invoke(question)

print("-" * 50)
print(f"최종 답변: {response}")
"""
print("\n" + "="*50)
print("RAG 질문-응답 시스템이 준비되었습니다.")
print("종료하려면 'q', 'quit', 또는 '종료'를 입력하세요.")
print("="*50)

while True:
    question = input("\n질문을 입력하세요: ").strip()

    if question.lower() in ['q', 'quit', 'exit', '종료']:
        print("프로그램을 종료합니다.")
        break

    if not question:
        print("질문을 입력해주세요.")
        continue

    response = chain.invoke(question)
    print("\n응답 메시지:", response)