from django.contrib import admin
from .models import Keyword, Lecture
# 모델을 Django Admin에서 사용할 수 있도록 등록
admin.site.register(Lecture)
admin.site.register(Keyword)

