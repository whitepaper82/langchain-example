from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# LangChain Community에서 OllamaEmbeddings를 가져옵니다.
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


loader = TextLoader('./data/history.txt', encoding="utf-8")
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=50,
    encoding_name='cl100k_base'
)

texts = text_splitter.split_text(data[0].page_content)
print(texts[0])

embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_texts(
    texts, 
    embeddings_model,
    collection_name = 'history',
    persist_directory = './db/chromadb',
    collection_metadata = {'hnsw:space': 'cosine'}, # l2 is the default
)

query = '누가 한글을 창제했나요?'
docs = db.similarity_search(query)
print(docs[0].page_content)