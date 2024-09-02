import os
import os.path as path
import platform
from datetime import datetime

# 환경변수 설정
from dotenv import load_dotenv

load_dotenv(".env")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_cookies_controller import CookieController

controller = CookieController()
try:
    controller.remove("selected_user_name")
    controller.remove("selected_loan_type")
    controller.remove("selected_amount_int")
except:
    pass

from boto3 import client
from matplotlib import font_manager, rc
from utils import APIHelper

# API Helper
api = APIHelper(
    api_url=os.environ.get("INFERENCE_API_URL"),
    api_key=os.environ.get("INFERENCE_API_KEY"),
)

# 데이터셋 csv 파일 다운로드
DATA_PATH = ".cache"
demo_path = path.join(DATA_PATH, "dataset_demo.csv")
total_path = path.join(DATA_PATH, "dataset_total.csv")

if not path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH, exist_ok=True)

if not path.isfile(demo_path):
    s3 = client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION"),
    )
    s3.download_file(
        "hktoss-mlops",
        "datasets/dataset_demo.csv",
        demo_path,
    )
    s3.close()

if not path.isfile(total_path):
    s3 = client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION"),
    )
    s3.download_file(
        "hktoss-mlops",
        "datasets/dataset_total.csv",
        total_path,
    )
    s3.close()

# HTML을 사용하여 스타일 추가
st.markdown(
    """
    <style>
    /* 사이드바 배경색 설정 */
    [data-testid="stSidebar"] {
        background-color: #0064FF;  /* Toss Blue */
    }

    /* 메인 화면 배경색 설정 */
    .css-18e3th9 {
        background-color: #202632;  /* Toss Gray */
    }

    /* 텍스트 및 위젯 스타일 설정 */
    .css-1lcbmhc, .css-14xtw13 {
        color: white;  /* 텍스트 색상 */
    }

    /* 드롭다운 클릭 시 테두리 색상 변경 */
    .stSelectbox > div > div {
        border-color: white !important;  /* 테두리 색상을 흰색으로 변경 */
    }

    /* 슬라이더 스타일 설정 */
    .css-16ws1b0 a {
        background-color: #0064FF !important;  /* Toss Blue */
    }

    /* 버튼 스타일 설정 */
    button[kind="secondary"] {
        color: white;  /* 버튼 텍스트 색상 */
        border: 2px solid white !important;  /* 버튼 테두리 색상 */
        border-radius: 5px;  /* 버튼 테두리 둥글기 */
    }
    a[data-testid='stSidebarNavLink'] {
        display:none;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# HTML을 사용한 글씨 사이즈 조정: sub-subheader 글씨 사이즈 추가 (기존 subheader 글씨 크기: 24px)
st.markdown(
    """
    <style>

    .font-size-sub-subheader {
        font-size:20px !important;
        font-weight: bold !important; /* 글씨체를 굵게 설정 */
    }

    .font-size-sub-write {
        font-size:14px !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# OS 별 폰트 깨짐 처리
if platform.system() == "Darwin":  # 맥
    plt.rc("font", family="AppleGothic")
elif platform.system() == "Windows":  # 윈도우
    plt.rc("font", family="Malgun Gothic")
elif platform.system() == "Linux":  # 리눅스 (구글 콜랩)
    # import matplotlib.font_manager as fm
    # fm._rebuild()
    plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False  # 한글 폰트 사용시 마이너스 폰트 깨짐 해결


# 데이터셋 불러오기
@st.cache_data
def load_demo_data():
    df = pd.read_csv(demo_path, low_memory=False)
    return df


def load_total_data():
    df = pd.read_csv(total_path, low_memory=False)
    return df


# 데이터 로드
demo = load_demo_data()
total = load_total_data()

# 열 이름 매핑
column_mapping = {
    "Credit_Utilization_Ratio": "대출 상환 비율",
    "Debt_to_Income_Ratio": "부채 상환 비율",
    "OVERDUE_RATIO": "대출 대비 연체 횟수 비율",
    "Debt_Repayment_Capability_Index": "부채 상환 가능성 지수",
    "LOAN_COUNT": "과거 대출 횟수",
    "AMT_CREDIT": "현재 대출 금액",
    "NAME": "이름",
    "DAYS_BIRTH": "나이",
    "CODE_GENDER": "성별",
    "FLAG_MOBIL": "휴대전화 소유 여부",
    "FLAG_OWN_CAR": "자차 소유 여부",  # 열 이름 확인 후 올바르게 수정
    "FLAG_OWN_REALTY": "부동산 소유 여부",
    "DAYS_EMPLOYED": "재직 여부",
    "AMT_INCOME_TOTAL": "연간 소득",
    "LOAN_STATUS": "대출 상태",
}
column_mapping_reverse = {v: k for (k, v) in column_mapping.items()}


def preprocess_api_input(selected_user):
    return (
        pd.DataFrame(selected_user).transpose().rename(columns=column_mapping_reverse)
    )


# 컬럼명 매핑 적용
demo.rename(columns=column_mapping, inplace=True)

# 대출 상품 딕셔너리 정의
loan_types = {
    "Consumer loans": "신용대출",
    "Cash loans": "현금대출",
    "Revolving loans": "리볼빙대출",
    "Mortgage": "주택담보대출",
    "Car loan": "자동차대출",
    "Microloan": "소액/비상금대출",
}

# 버튼 상태 초기화
st.session_state.predict_clicked = False

st.sidebar.page_link(page="./app.py", label="Home", icon="🏠")

with st.sidebar.form(key="sidebar_form"):

    name = st.selectbox("이름을 선택하세요", demo["이름"].unique())

    # 빈 공간 추가
    st.write(" ")
    st.write(" ")

    selected_loan_type = st.selectbox(
        "대출상품을 선택하세요",
        options=list(loan_types.keys()),
        format_func=lambda x: loan_types[x],
    )

    # 빈 공간 추가
    st.write(" ")
    st.write(" ")

    # text_input을 사용하여 대출 금액 직접 입력

    selected_amount = st.text_input(
        "대출 금액 입력 (예: 1, 2 ,..., 50)", value="1"  # 기본값 설정
    )

    st.markdown(
        "<p class='font-size-sub-write'>" "[단위: 천만원]" "</p>",
        unsafe_allow_html=True,
    )

    st.write()

    try:
        selected_amount_int = 10000000 * int(selected_amount)
        if not (10_000_000 <= selected_amount_int <= 500_000_000):
            st.write(
                "대출 금액이 유효한 범위 내에 있지 않습니다. 1천만원에서 50천만원 사이로 입력하세요."
            )
    except ValueError:
        st.write("유효한 숫자를 입력하세요.")

    # '확인하기' 버튼 클릭 상태를 세션 상태에 저장
    predict_button = st.form_submit_button("확인하기")
    if predict_button:
        st.session_state.predict_clicked = True

        # Set Cookie - Predict button submit
        controller.set("selected_amount_int", selected_amount_int)
        controller.set("selected_loan_type", selected_loan_type)
        controller.set("selected_user_name", name)

# 본 화면

if "show_more" not in st.session_state:
    st.session_state.show_more = False

# 조건을 설정한 후 확인하기를 눌러주세요 문구 추가
if not st.session_state.predict_clicked:
    st.markdown(
        "<p style='text-align: center; color: rgba(255, 255, 255, 0.5); font-size: 20px;'>        좌측 사이드바에서 조건을 설정한 후 확인하기를 눌러주세요</p>",
        unsafe_allow_html=True,
    )

# '확인하기' 버튼이 눌렸는지 확인
if st.session_state.predict_clicked:
    st.title(f"{name}님의 대출 가능성 분석")

    selected_user = demo[demo["이름"] == name]

    if selected_user.empty:
        st.write("선택된 사용자 데이터가 없습니다.")
    else:
        selected_user = selected_user.iloc[0]

        def calculate_age(days_birth):
            today = datetime.today()
            birth_date = today - pd.to_timedelta(-days_birth, unit="D")
            age = today.year - birth_date.year
            if today.month < birth_date.month or (
                today.month == birth_date.month and today.day < birth_date.day
            ):
                age -= 1
            return age

        # 점선 추가
        st.markdown("<hr style='border: 1px dashed gray;' />", unsafe_allow_html=True)

        age = calculate_age(selected_user["나이"])

        st.subheader("현재 나의 정보")
        st.markdown(f"**😀 나이:** {age} 세")
        st.markdown(f"**👫 성별:** {selected_user['성별']}")  # 성별 추가
        st.markdown(
            f"**📱 휴대전화 소유 여부:** {'Y' if selected_user['휴대전화 소유 여부'] else 'N'}"
        )
        st.markdown(
            f"**🚗 자차 소유 여부:** {'Y' if selected_user['자차 소유 여부'] else 'N'}"
        )
        st.markdown(
            f"**🏡 부동산 소유 여부:** {'Y' if selected_user['부동산 소유 여부'] else 'N'}"
        )
        st.markdown(f"**🏢 재직 여부:** {'Y' if selected_user['재직 여부'] else 'N'}")

        # 점선 추가
        st.markdown("<hr style='border: 1px dashed gray;' />", unsafe_allow_html=True)

        # 신청한 대출 정보와 평가 버튼 추가
        st.subheader("신청한 대출 정보")
        st.write(
            f"🏦 **선택한 대출 상품:** {loan_types[selected_loan_type]}"
        )  # 선택한 대출 상품 표시
        st.write(
            f"💵 **선택한 대출 금액:** ₩{selected_amount_int:,}원"
        )  # 선택한 대출 금액 표시

        st.write(" ")

        st.page_link("pages/analysis.py", label="💰 대출 가능성 분석 💰")
