from django.shortcuts import render, redirect
from django.utils import timezone
from django.db.models import Q
from django.apps import apps
from django.core.cache import cache
from dotenv import load_dotenv
from .models import Lecture, Keyword
from konlpy.tag import Okt
from .stopwords import STOP_WORDS, JOSA, PUNCTUATION, COMPOUND_NOUNS
from langchain_openai import ChatOpenAI
from langchain_community.utils.math import cosine_similarity
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_community.retrievers import TFIDFRetriever
import numpy as np
import pytz
import time
import os
import hashlib

# env 불러오기
load_dotenv()

# 임베딩
def embed_question_with_huggingface(question):
    embeddings = apps.get_app_config('Chatbot').embeddings
    if embeddings is None:
        raise RuntimeError("Embeddings가 아직 초기화되지 않았습니다")
    
    embedding_vector = embeddings.embed_query(question)
    return np.array(embedding_vector)

def find_similar_question_with_huggingface(question):
    cached_questions = list(cached_keys)

    if not cached_questions:
        return None

    try:
        input_question_embedding = embed_question_with_huggingface(question)
        cached_question_embeddings = np.array([
            embed_question_with_huggingface(cached_question)
            for cached_question in cached_questions
        ])

        similarities = cosine_similarity(
            input_question_embedding.reshape(1, -1),
            cached_question_embeddings.reshape(len(cached_questions), -1)
        ).flatten()

        max_similarity_index = similarities.argmax()
        max_similarity = similarities[max_similarity_index]

        if max_similarity > 0.7:
            similar_question = cached_questions[max_similarity_index]
            return cache.get(similar_question)

        return None

    except Exception as e:
        print(f"일반 오류: {e}")
        return "오류가 발생했습니다."

def correct_question_with_huggingface(question):
    cache_key = generate_cache_key(question)
    cached_response = find_similar_question_with_huggingface(question)

    if cached_response:
        print(f"[디버깅] 캐시된 유사한 질문을 찾았습니다: {cached_response}")
        return cached_response

    corrected_question = question
    print(f"[디버깅] 교정 전 질문: {question}")
    print(f"[디버깅] 교정 후 질문: {corrected_question}")

    keyword_kor = extract_nouns_and_adjectives_korean(corrected_question)

    cache.set(cache_key, corrected_question, timeout=3600)
    cached_keys.add(cache_key)

    return corrected_question

okt = Okt()
cached_keys = set()

def generate_cache_key(question):
    return hashlib.md5(question.encode("utf-8")).hexdigest()

def find_similar_question(question):
    cached_questions = list(cached_keys)

    if not cached_questions:
        return None

    try:
        # TFIDFRetriever 초기화
        retriever = TFIDFRetriever.from_texts(
            cached_questions,
            k=1  # 가장 유사한 1개의 문서만 반환
        )

        # 가장 유사한 문서 검색
        similar_docs = retriever.get_relevant_documents(question)

        if similar_docs:
            # 첫 번째(가장 유사한) 문서의 내용을 가져옴
            similar_question = similar_docs[0].page_content
            similarity_score = similar_docs[0].metadata.get('score', 0)

            # 유사도가 0.7 이상인 경우에만 캐시된 응답 반환
            if similarity_score > 0.7:
                return cache.get(similar_question)

        return None

    except Exception as e:
        print(f"유사 질문 검색 중 오류 발생: {e}")
        return None

def fetch_openai_response(question):
    try:
        chat = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "당신은 대한민국 초등학교 4학년 수학 용어를 전문적으로 교정하는 도우미입니다. "
                "제공된 용어에서 모든 철자 및 오타를 최대한 정확하게 교정하세요. "
                "오타나 철자 오류가 있는 용어도 최대한 추론하여 올바른 용어로 수정하세요. "
                "그 후, 교정된 용어 중 초등학교 4학년 수학 교과서에 나오는 핵심 용어만 추출해 주세요. "
                "일반적인 단어는 제외하고, 수학과 직접적으로 관련된 용어만 포함하세요. "
                "추가적인 단어를 포함하지 말고, 교정된 용어만 출력하세요."
            ),
            HumanMessagePromptTemplate.from_template(
                "질문의 오타를 모두 교정한 후, 초등학교 4학년 수학과 관련된 올바른 용어만 추출해 주세요. "
                "추가적인 단어를 포함하지 말고, 교정된 질문에 등장하는 정확한 키워드만 추출하세요. "
                "교정된 키워드만 출력해주세요. 질문은: {question} 입니다."
            )
        ])

        chain = prompt | chat
        response = chain.invoke({"question": question})
        return response.content.strip()

    except Exception as e:
        print(f"오류 발생: {e}")
        return "오류가 발생했습니다. 다시 시도해 주세요."

def extract_nouns_and_adjectives_korean(question):
    start_time = time.time()

    found_compounds = []
    for compound in COMPOUND_NOUNS:
        if compound in question:
            found_compounds.append(compound)

    tokens = okt.pos(question, norm=True, stem=True)
    keywords = [word for word, pos in tokens if pos in ["Noun", "Adjective"]]

    filtered_keywords = [
        keyword for keyword in keywords if keyword not in STOP_WORDS
        and keyword not in JOSA and keyword not in PUNCTUATION
    ]

    if found_compounds:
        filtered_keywords.extend(found_compounds)

    unique_keywords = list(set(filtered_keywords))
    unique_keywords = [keyword.replace(" ", "") for keyword in unique_keywords]

    end_time = time.time()
    print(f"실행 시간: {end_time - start_time} 초")

    return unique_keywords

def get_lecture_data(keywords):
    keyword_objects = Keyword.objects.filter(keyword__in=keywords)
    lectures = Lecture.objects.filter(keywords__in=keyword_objects).distinct()
    print(f"검색된 강의 수: {lectures.count()}")
    return lectures


def chatbot_response(question):
    start_time = time.time()

    keyword_kor = extract_nouns_and_adjectives_korean(question)
    results = get_lecture_data(keyword_kor)

    if not results.exists():
        corrected_question = correct_question_with_huggingface(question)
        keyword_kor = extract_nouns_and_adjectives_korean(corrected_question)
        results = get_lecture_data(keyword_kor)

        if not results.exists():
            return ["죄송합니다. 강의를 찾지 못했습니다."]

    end_time = time.time()
    print(f"chatbot_response 실행 시간: {end_time - start_time} 초")

    return list(results)

# 채팅 기록 화면
def qa_process(request):
    results = None
    chat_history = request.session.get("chat_history", [])
    show_cards = False

    if request.method == "POST":
        question = request.POST.get("question_input")
        results = chatbot_response(question)
        
        chatbot_config = apps.get_app_config('Chatbot')
        retriever = chatbot_config.retriever
        
        print(retriever.invoke("{question}"))

        kst = pytz.timezone("Asia/Seoul")
        current_time = timezone.now().astimezone(kst).strftime("%Y-%m-%d %H:%M:%S")

        if results and isinstance(results[0], Lecture):
            show_cards = True
            answer_str = ", ".join([result.lecture_title for result in results])
        else:
            answer_str = results[0]

        chat_history.append({
            "question": question,
            "answer": answer_str,
            "timestamp": current_time,
        })
        request.session["chat_history"] = chat_history

    keywords = [
        "큰 수", "곱셈", "사각형", "꺾은선그래프", "소수", "분수", "다각형",
        "평면도형", "막대그래프", "각도", "규칙 찾기", "삼각형", "나눗셈",
    ]

    context = {
        "results": results,
        "chat_history": chat_history,
        "show_cards": show_cards,
        "keywords": keywords,
    }

    return render(request, "qa_template.html", context)

# 채팅 기록 삭제
def clear_history(request):
    request.session.pop("chat_history", None)
    return redirect("qa_process")
