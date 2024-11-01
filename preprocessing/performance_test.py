import csv
import sys
import os
import django
import json
import time
import difflib

# 현재 스크립트의 상위 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Django 설정 파일을 환경 변수로 설정
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Chatbot.settings")

# Django 환경 초기화
django.setup()

from Chatbot.views import chatbot_response


def load_test_data(csv_file_path):
    test_data = []
    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            expected_value = row.get("expected",
                                     "")  # "expected" 필드가 없으면 빈 문자열
            if expected_value:
                expected_value = expected_value.replace("'", '"')
                try:
                    expected_keywords = json.loads(expected_value)
                except json.JSONDecodeError:
                    print(f"JSONDecodeError: {expected_value}")  # 오류 발생 시 출력
                    expected_keywords = []  # 기본값으로 빈 리스트 설정
            else:
                expected_keywords = []  # "expected" 값이 없으면 빈 리스트로 설정

            test_data.append({
                "question":
                row.get("question",
                        "Unknown question"),  # "question" 필드가 없을 경우 기본값 설정
                "expected_keywords":
                expected_keywords,
            })
    return test_data


def evaluate_chatbot_performance(test_data):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    correct_predictions = 0  # 정확히 예측한 질문의 수를 세기 위한 변수
    total_questions = len(test_data)
    total_response_time = 0  # 총 응답 시간을 저장할 변수

    # CSV 파일 열기
    with open("similarity_results.csv", mode="w", newline="",
              encoding="utf-8") as csvfile:
        fieldnames = ["유사도", "예상 키워드", "예측된 강의 제목"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for data in test_data:
            question = data["question"]
            expected_keywords = set(
                data["expected_keywords"])  # 이미 리스트로 변환된 상태

            # 응답 시간 측정 시작
            start_time = time.time()

            # 실제 챗봇 검색 함수 호출
            predicted_lectures = chatbot_response(question)  # 반환된 강의 리스트

            # 응답 시간 측정 종료
            end_time = time.time()
            total_response_time += end_time - start_time  # 응답 시간 누적

            # Lecture 객체인지 확인하고, Lecture 객체일 경우 강의 제목 추출
            predicted = set()
            for lecture in predicted_lectures:
                if hasattr(lecture, "lecture_title"):
                    predicted.add(
                        lecture.lecture_title)  # Lecture 객체인 경우 제목 추출
                else:
                    predicted.add(lecture)  # Lecture 객체가 아니면 문자열 그대로 추가

            print(f"예상 키워드: {expected_keywords}")
            print(f"예측된 강의 제목: {predicted}")

            # 유사도 기준으로 True Positive 계산
            for expected_keyword in expected_keywords:
                for predicted_title in predicted:
                    similarity = difflib.SequenceMatcher(
                        None, expected_keyword, predicted_title).ratio()
                    print(
                        f"유사도: {similarity}, 예상 키워드: {expected_keyword}, 예측된 강의 제목: {predicted_title}"
                    )
                    writer.writerow({
                        "유사도": similarity,
                        "예상 키워드": expected_keyword,
                        "예측된 강의 제목": predicted_title,
                    })

                    if similarity >= 0.5:  # 유사도가 0.5 이상인 경우
                        true_positive += 1
                        break  # 하나의 유사도가 만족하면 더 이상 확인할 필요 없음

            # False Positive
            false_positive += len(predicted - expected_keywords)

            # False Negative
            false_negative += len(expected_keywords - predicted)

            # 전체가 맞는지 확인하여 correct_predictions 증가
            if predicted == expected_keywords:
                correct_predictions += 1

    # True Negative 계산
    true_negative = total_questions - (true_positive + false_positive +
                                       false_negative)

    # 정밀도(Precision)
    precision = (true_positive / (true_positive + false_positive) if
                 (true_positive + false_positive) > 0 else 0)

    # 재현율(Recall)
    recall = (true_positive / (true_positive + false_negative) if
              (true_positive + false_negative) > 0 else 0)

    # F1 Score
    f1_score = (2 * (precision * recall) / (precision + recall) if precision +
                recall > 0 else 0)

    # 정확도: 전체 질문 중에서 정확히 예측된 질문의 비율
    accuracy = correct_predictions / total_questions if total_questions > 0 else 0

    # 평균 응답 시간 계산
    average_response_time = (total_response_time /
                             total_questions if total_questions > 0 else 0)

    # 성능 결과 출력
    print(f"정밀도(Precision): {precision:.2f}")
    print(f"재현율(Recall): {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print(f"정확도(Accuracy): {accuracy:.2f}")
    print(f"평균 응답 시간: {average_response_time:.4f} 초")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_dir,
                                 "corrected_unique_test_questions.csv")
    test_data = load_test_data(csv_file_path)
    evaluate_chatbot_performance(test_data)
