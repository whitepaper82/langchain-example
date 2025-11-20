### PyPDFLoader

from langchain_community.document_loaders import PyPDFLoader

pdf_filepath = './data/000660_SK_2023.pdf'
loader = PyPDFLoader(pdf_filepath)
pages = loader.load()

print("\nPyPDFLoader len: ",len(pages))
#print(pages[1])


### UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader

pdf_filepath = './data/000660_SK_2023.pdf'

# 전체 텍스트를 단일 문서 객체로 변환
loader = UnstructuredPDFLoader(pdf_filepath)
pages = loader.load()

print("\nUnstructuredPDFLoader len: ",len(pages))

### OnlinePDF Loader
from langchain_community.document_loaders import OnlinePDFLoader

# Transformers 논문을 로드
loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
pages = loader.load()

print("\nOnlinePDFLoader len : ",len(pages))

#로드된 문서 객체의 내용을 출력하여 확인합니다. 첫 페이지의 텍스트 내용 중 처음 1000자를 출력합니다.
#print(pages[0].page_content[:1000])