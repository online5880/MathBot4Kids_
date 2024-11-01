from django.apps import AppConfig
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os

class SttConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "Chatbot"
    
    model_name = "nlpai-lab/KoE5"
    embeddings = None

    def ready(self):
        import sys
        
        if 'runserver' not in sys.argv:
            return
            
        from .views import extract_nouns_and_adjectives_korean
        
        if os.environ.get('RUN_MAIN'):  # 메인 프로세스에서만 실행
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            print("서버 시작 시 extract_nouns_and_adjectives_korean 실행")
            test_question = "서버 실행합니다."
            keywords = extract_nouns_and_adjectives_korean(test_question)
            print(f"추출된 키워드: {keywords}")
