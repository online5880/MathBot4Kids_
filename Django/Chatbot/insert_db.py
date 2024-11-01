import sys
import os
import django
import pandas as pd

# 현재 스크립트의 상위 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Django 설정 파일을 환경 변수로 설정
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Chatbot.settings")

# Django 환경 초기화
django.setup()

from Chatbot.models import Lecture, Keyword

# 요약 데이터를 업데이트할 폴더 경로 설정
summary_folders = ["summary_list/4_1", "summary_list/4_2"]

# 요약 데이터 업데이트 함수
def update_summary_from_csv():
    # CSV 파일에서 "요약" 데이터를 모아둠
    summary_data = {}

    # summary 폴더 내 모든 CSV 파일 처리
    for folder in summary_folders:
        for file_name in os.listdir(folder):
            print(file_name)
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder, file_name)
                csv_data = pd.read_csv(file_path)

                # '강의 제목'을 기준으로 요약 데이터를 매핑 (강의 제목은 유일하다고 가정)
                for index, row in csv_data.iterrows():
                    lecture_title = row["강의 제목"]
                    lecture_summary = row["요약"]

                    # 데이터 확인을 위해 출력해봅니다.
                    print(f"강의 제목: {lecture_title}, 요약: {lecture_summary}")
                    summary_data[lecture_title] = lecture_summary

    # 데이터베이스의 Lecture 테이블에서 "요약"만 업데이트
    lectures = Lecture.objects.all()
    for lecture in lectures:
        if lecture.lecture_title in summary_data:
            lecture.lecture_summary = summary_data[lecture.lecture_title]
            lecture.save()  # 변경된 내용을 저장

    print("요약 데이터를 성공적으로 업데이트했습니다.")

# 기존 데이터 삽입 로직 (강의 및 키워드 추가)
def insert_lecture_data():
    # CSV 파일 읽기 (기본 데이터)
    df = pd.read_csv("final_data_modify.csv")

    # 데이터베이스에 데이터 삽입
    for index, row in df.iterrows():
        # Lecture 객체 생성 및 저장
        lecture = Lecture.objects.create(
            lecture_title=row["강의 제목"],
            lecture_content=row["강의 내용"],
            lecture_summary=row["요약"],  # 나중에 덮어씌워질 부분
            thumbnail_url=row["썸네일 URL"],
            semester=row["강의 학기"],  # 4-1 또는 4-2 정보
        )
        
        # 키워드 추가 (여러 키워드가 있는 경우 쉼표로 구분된 것으로 가정)
        keywords = row["키워드"].split(",")  # '키워드'라는 컬럼이 있다고 가정
        for keyword in keywords:
            keyword_obj, created = Keyword.objects.get_or_create(
                keyword=keyword.strip())
            lecture.keywords.add(keyword_obj)  # Many-to-Many 관계 추가

    # 완료 메시지
    print("기본 데이터를 성공적으로 삽입했습니다.")
df = pd.read_csv("final_data_modify.csv")
df['요약'][0]
# 요약 데이터 업데이트 실행
update_summary_from_csv()
