from langchain_community.document_loaders import TextLoader

loader = TextLoader('./data/history.txt', encoding="utf-8")
data = loader.load()

print(type(data))
print(len(data))
print(data)


###### Directory Loader
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader(path='./data/', glob='*.txt', loader_cls=TextLoader,loader_kwargs={'encoding': 'utf-8'})
data = loader.load()
print(f"\n\nDiroectory 로드된 문서 개수: {len(data)}")


###### CSV Loader
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='./data/주택금융관련_지수_20160101.csv', encoding='cp949')
data = loader.load()
print(len(data))

print("1 column\n",data[0])
print("\n2 column\n",data[1])