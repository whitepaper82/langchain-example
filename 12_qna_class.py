import os
import langchain
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from transformers import AutoTokenizer
from langchain_classic import hub
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

class QnAAgent:
    """
    PDF 문서를 로드하고 질문에 답변하는 Q&A 에이전트 클래스입니다.
    """
    
    def __init__(self, pdf_path: str = './data/SPRi AI Brief_11월호_산업동향_1105_F.pdf', model_id: str = "google/gemma-3-12b-it"):
        """
        클래스를 초기화하고 문서 로드, 임베딩, 벡터 저장소 생성을 수행합니다.
        
        :param pdf_path: 로드할 PDF 파일 경로
        :param model_id: 토크나이저 모델 ID
        """
        load_dotenv()
        
        self.pdf_path = pdf_path
        self.model_id = model_id
        
        # 1. 토크나이저 설정
        self._setup_tokenizer()
        
        # 2. LLM 설정
        self.llm = ChatOllama(
            base_url="http://localhost:11434",
            model="gemma3:1b"
        )
        
        # 3. 임베딩 모델 설정
        self.embeddings = HuggingFaceEmbeddings(
            model_name='jhgan/ko-sbert-nli',
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings':True},
        )
        
        # 4. RAG 파이프라인 구축 (문서 로드 -> 분할 -> DB 저장 -> 검색기)
        self.retriever = self._build_rag_pipeline()
        
        # 5. 프롬프트 및 체인 설정
        self.chain = self._build_chain()
        
        print(f"✅ QnAAgent 초기화 완료. 대상 문서: {self.pdf_path}")

    def _setup_tokenizer(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            print(f"'{self.model_id}' 토크나이저 로드 성공")
        except Exception as e:
            print(f"⚠️ 토크나이저 로드 실패. 대체 모델(gpt2)을 사용합니다.\n오류: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text))

    def _build_rag_pipeline(self):
        # 단계 1: 문서 로드
        if not os.path.exists(self.pdf_path):
             raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.pdf_path}")
             
        loader = PyMuPDFLoader(self.pdf_path)
        data = loader.load()
        
        # 단계 2: 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        split_documents = text_splitter.split_documents(data)
        print(f"문서 수: {len(data)}")
        print(f"분할된 청크의수: {len(split_documents)}")
        
        # 단계 3: DB 생성 및 저장
        # 참고: 매번 새로 생성하는 로직을 그대로 유지합니다.
        vectorstore = Chroma.from_documents(
            split_documents, 
            self.embeddings,
            collection_name = 'SPRi',
            persist_directory = './db/chromadb',
            collection_metadata = {'hnsw:space': 'cosine'},
        )
        
        # 단계 4: 검색기 생성
        return vectorstore.as_retriever(search_kwargs={'k': 3})

    def _build_chain(self):
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
        
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def answer(self, question: str) -> str:
        """
        질문에 대한 답변을 생성합니다.
        
        :param question: 질문 내용
        :return: 답변 문자열
        """
        print(f"\n--- [질문] '{question}' ---")
        response = self.chain.invoke(question)
        return response

    def get_retriever(self):
        """검색기(Retriever) 인스턴스를 반환합니다."""
        return self.retriever

# --- 사용 예시 ---
if __name__ == "__main__":
    try:
        # 1. 에이전트 초기화
        agent = QnAAgent()
        
        # 2. 질문하기
        question = "구글의 최신 동영상 생성 AI 모델 이름은?"
        response = agent.answer(question)
        
        print("\n\n응답 메시지 : ", response)
        
        # 3. (옵션) 검색된 문서 확인
        print("\n=== 검색된 문서 내용 확인 (디버깅) ===")
        retriever = agent.get_retriever()
        docs = retriever.invoke(question)
        for i, doc in enumerate(docs):
            print(f"[문서 {i+1}] {doc.page_content[:100]}...")
        print("===========================")
        
    except Exception as e:
        print(f"\n❌ 에이전트 실행 중 오류 발생: {e}")
