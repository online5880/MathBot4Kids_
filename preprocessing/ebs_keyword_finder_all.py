from selenium.webdriver.common.by import By
from selenium import webdriver
import time
import pandas as pd

# ChromeDriver 설정
driver = webdriver.Chrome()

# 학년/학기별 EBS URL과 학년 정보를 함께 저장
urls = [
    {"url": "https://primary.ebs.co.kr/course/view?courseId=100004633&left=series", "학년": "1학년", "학기": "1학기"},
    {"url": "https://primary.ebs.co.kr/course/view?courseId=100005135&left=series", "학년": "1학년", "학기": "2학기"},
    {"url": "https://primary.ebs.co.kr/course/view?courseId=100004634&left=series", "학년": "2학년", "학기": "1학기"},
    {"url": "https://primary.ebs.co.kr/course/view?courseId=100005136&left=series", "학년": "2학년", "학기": "2학기"},
    {"url": "https://primary.ebs.co.kr/course/view?courseId=100004164&left=series", "학년": "3학년", "학기": "1학기"},
    {"url": "https://primary.ebs.co.kr/course/view?courseId=100004318&left=series", "학년": "3학년", "학기": "2학기"},
    {"url": "https://primary.ebs.co.kr/course/view?courseId=100004167&left=series", "학년": "4학년", "학기": "1학기"},
    {"url": "https://primary.ebs.co.kr/course/view?courseId=100004295&left=series", "학년": "4학년", "학기": "2학기"},
    {"url": "https://primary.ebs.co.kr/course/view?courseId=100004437&left=series", "학년": "5학년", "학기": "1학기"},
    {"url": "https://primary.ebs.co.kr/course/view?courseId=100004512&left=series", "학년": "5학년", "학기": "2학기"},
    {"url": "https://primary.ebs.co.kr/course/view?courseId=100004441&left=series", "학년": "6학년", "학기": "1학기"},
    {"url": "https://primary.ebs.co.kr/course/view?courseId=100004515&left=series", "학년": "6학년", "학기": "2학기"},
]

# 데이터를 저장할 리스트 준비
data = []

# 각 URL을 순회하면서 데이터 스크래핑
for item in urls:
    url = item["url"]
    학년 = item["학년"]
    학기 = item["학기"]
    
    driver.get(url)
    
    # 페이지가 완전히 로드될 때까지 잠시 대기 (필요에 따라 조정 가능)
    time.sleep(5)
    
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

            # 데이터를 리스트에 저장 (학년 및 학기 포함)
            data.append({
                "학년": 학년,
                "학기": 학기,
                "강의 제목": title_text,
                "수업내용": dt_text,
                "키워드": dd_text,
            })

    except Exception as e:
        print(f"데이터 추출에 실패했습니다 ({url}): {e}")

# WebDriver 종료
driver.quit()

# 추출한 데이터를 pandas 데이터프레임으로 변환 (컬럼명을 포함)
df = pd.DataFrame(data)

# 중복된 '키워드'가 있는 행 제거
df = df.drop_duplicates(subset="키워드", keep="first")

# 데이터프레임 출력
print(df)

# 데이터프레임을 파일로 저장할 경우
df.to_csv("강의_학년별_키워드_수업내용.csv", index=False)
