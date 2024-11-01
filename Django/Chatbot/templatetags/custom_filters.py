from django import template
import ast

register = template.Library()

# 문자열을 리스트로 변환하는 필터
@register.filter
def to_list(value):
    """ 문자열을 Python 리스트로 변환 """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []  # 변환 실패 시 빈 리스트 반환

# 문자열의 앞뒤 공백을 제거하는 필터
@register.filter
def strip_spaces(value):
    """ 문자열의 앞뒤 공백을 제거 """
    return value.strip() if isinstance(value, str) else value
