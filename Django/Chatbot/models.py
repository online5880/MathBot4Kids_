from django.db import models


class Keyword(models.Model):
    keyword = models.CharField(max_length=255)  # 키워드


class Lecture(models.Model):
    lecture_title = models.CharField(max_length=255)  # 강의 제목
    lecture_content = models.TextField()  # 강의 내용
    lecture_summary = models.TextField()  # 강의 요약
    thumbnail_url = models.URLField(max_length=255)  # 썸네일 이미지 URL
    semester = models.CharField(max_length=50)  # 강의 학기 (예: "2024-1학기")

    # Many-to-Many 관계 정의
    keywords = models.ManyToManyField(Keyword, related_name="lectures")
