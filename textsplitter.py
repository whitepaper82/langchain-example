from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader('./data/history.txt')
data = loader.load()

print(len(data[0].page_content))   # history.txt 내 문자열 수는 1234


# 각 문자를 구분하여 분할
text_splitter = CharacterTextSplitter(
    separator = '',
    chunk_size = 500,
    chunk_overlap  = 100,
    length_function = len,
)

texts = text_splitter.split_text(data[0].page_content)

print(len(texts))  # output : 3
print(len(texts[0])) # output : 500