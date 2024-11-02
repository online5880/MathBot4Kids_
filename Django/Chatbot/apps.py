from django.apps import AppConfig
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
import os

def load_csv():
    loader = CSVLoader("../data/final_data.csv")
    documents = loader.load()
    return documents

def text_split(documents):
    text_split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_data = text_split.split_documents(documents)
    return split_data

def embeddings(model_name="nlpai-lab/KoE5"):
    embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    return embeddings

def vectorstore(split_data,embeddings):
    vectorstore = FAISS.from_documents(documents=split_data, embedding=embeddings)
    vectorstore.save_local("vectorstore")
    return vectorstore

def init():
    path = "vectorstore"
    index_file = os.path.join(path, "index.faiss")
    _embedding = embeddings()  # 사용자 정의 임베딩 함수 호출

    if os.path.exists(index_file):
        # 벡터 스토어가 로컬에 존재하면 로드
        return FAISS.load_local(path, _embedding, allow_dangerous_deserialization=True)
    else:
        # 벡터 스토어가 존재하지 않으면 생성
        _docs = load_csv()                # CSV 파일 로드
        _text = text_split(_docs)          # 문서 분할
        _vectorstore = FAISS.from_documents(_text, _embedding)
        
        # 생성된 벡터 스토어를 로컬에 저장
        _vectorstore.save_local(path)
        return _vectorstore

    

class SttConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "Chatbot"
    
    retriever = None

    def ready(self):
        import sys
        
        if 'runserver' not in sys.argv:
            return
            
        from .views import extract_nouns_and_adjectives_korean
        
        if os.environ.get('RUN_MAIN'):  # 메인 프로세스에서만 실행
            
            vectorstore = init()
            if vectorstore:
                self.retriever = vectorstore.as_retriever()
                print("검색기 생성 완료")    
            

            print("서버 시작 시 extract_nouns_and_adjectives_korean 실행")
            test_question = "서버 실행합니다."
            keywords = extract_nouns_and_adjectives_korean(test_question)
            print(f"추출된 키워드: {keywords}")
