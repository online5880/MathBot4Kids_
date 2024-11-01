from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
import time
import pandas as pd

# ChromeDriver 설정
driver = webdriver.Chrome()

# EBS URL
# https://primary.ebs.co.kr/course/view?courseId=100004633&left=series # 1학년 1학기
# https://primary.ebs.co.kr/course/view?courseId=100005135&left=series # 1학년 2학기
# https://primary.ebs.co.kr/course/view?courseId=100004634&left=series # 2학년 1학기
# https://primary.ebs.co.kr/course/view?courseId=100005136&left=series # 2학년 2학기
# https://primary.ebs.co.kr/course/view?courseId=100004164&left=series # 3학년 1학기
# https://primary.ebs.co.kr/course/view?courseId=100004318&left=series # 3학년 2학기
# https://primary.ebs.co.kr/course/view?courseId=100004167&left=series  # 4학년 1학기
# https://primary.ebs.co.kr/course/view?courseId=100004295&left=series  # 4학년 2학기
# https://primary.ebs.co.kr/course/view?courseId=100004437&left=series # 5학년 1학기
# https://primary.ebs.co.kr/course/view?courseId=100004512&left=series # 5학년 2학기
# https://primary.ebs.co.kr/course/view?courseId=100004441&left=series # 6학년 1학기
# https://primary.ebs.co.kr/course/view?courseId=100004515&left=series # 6학년 2학기

# 암시적 대기 (최대 30초까지 대기)
driver.implicitly_wait(30)

# URL 로드
driver.get(url)

# 페이지가 완전히 로드될 때까지 기다림
time.sleep(5)  # 페이지 로딩 시간을 충분히 늘림

# 데이터를 저장할 리스트 준비
data = []

try:
    # '강의 제목'이 포함된 strong 태그들을 추출
    title_elements = driver.find_elements(
        By.XPATH, "//div[@class='wrap_webtoon']//strong")

    # '강의설명'과 '키워드'가 포함된 dt와 dd 태그들을 추출
    lecture_elements = driver.find_elements(By.XPATH,
                                            "//div[@class='con_wrap']")

    # 강의 제목, 설명, 키워드를 추출하여 데이터를 추가
    for i, container in enumerate(lecture_elements):
        # 강의 제목 추출 (강의 제목을 못 찾으면 N/A)
        title_text = (title_elements[i].text.strip()
                      if i < len(title_elements) else "N/A")

        # 강의 설명 추출 (강의 설명을 못 찾으면 기본값 N/A)
        try:
            dt_text = container.find_element(
                By.XPATH,
                ".//dt[text()='강의설명']/following-sibling::dd").text.strip()
        except Exception as e:
            dt_text = "N/A"

        # 키워드 추출 (키워드를 못 찾으면 기본값 N/A)
        try:
            dd_text = container.find_element(
                By.XPATH,
                ".//dt[text()='키워드']/following-sibling::dd").text.strip()
        except Exception as e:
            dd_text = "N/A"

        # 썸네일 이미지 추출
        try:
            thumbnail_element = container.find_element(
                By.XPATH, ".//p[@class='sub_thumb']/img")
            thumbnail_url = thumbnail_element.get_attribute("src")
        except Exception as e:
            thumbnail_url = "N/A"

        # 데이터를 리스트에 저장
        data.append({
            "강의 제목": title_text,
            "수업내용": dt_text,
            "키워드": dd_text,
            "썸네일 URL": thumbnail_url,
        })

except Exception as e:
    print(f"데이터 추출에 실패했습니다: {e}")

# WebDriver 종료
driver.quit()

# 추출한 데이터를 pandas 데이터프레임으로 변환 (컬럼명을 포함)
df = pd.DataFrame(data)

# 중복된 '키워드'가 있는 행 제거
# df = df.drop_duplicates(subset="키워드", keep="first")

# 데이터프레임 출력
print(df)

# 데이터프레임을 파일로 저장할 경우
df.to_csv("강의_키워드_썸네일_데이터.csv", index=False)
