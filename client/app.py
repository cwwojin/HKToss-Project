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

    </style>
    """,
    unsafe_allow_html=True,
)

# font_name = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font_name)

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

    # # 옵션 리스트를 문자열 형식으로 생성
    # options = [f"₩{x:,}" for x in range(1_000_000, 50_000_000, 1_000_000)]

    # # select_slider를 사용하여 사용자 정의 슬라이더 구현
    # credit_min = st.select_slider(
    #     "대출 금액 선택 (최대값: ₩50,000,000은 선택 불가)",
    #     options=options,  # 포맷된 옵션 사용
    #     value="₩1,000,000",  # 기본값 설정
    # )

    # # # 선택된 금액 출력
    # # st.write(f"선택한 대출 금액: {credit_min}")

    # # '확인하기' 버튼을 추가하여 연체 예측 결과를 확인
    # predict_button = st.form_submit_button("확인하기")

    # text_input을 사용하여 대출 금액 직접 입력
    selected_amount = st.text_input(
        "대출 금액 입력 (예: 1000000)", value="1000000"  # 기본값 설정
    )

    try:
        selected_amount_int = int(selected_amount)
        if not (1_000_000 <= selected_amount_int <= 50_000_000):
            st.write(
                "대출 금액이 유효한 범위 내에 있지 않습니다. 1,000,000원에서 50,000,000원 사이로 입력하세요."
            )
    except ValueError:
        st.write("유효한 숫자를 입력하세요.")

    # '확인하기' 버튼을 추가하여 연체 예측 결과를 확인
    predict_button = st.form_submit_button("확인하기")

# 본 화면


# 시각화 함수 정의
def create_style(ax):
    fig.patch.set_facecolor("#0E1117")  # 전체 figure 배경 색상 설정
    ax.set_facecolor("#0E1117")  # 개별 subplot 배경 색상 설정
    ax.spines["top"].set_color("#31333F")
    ax.spines["right"].set_color("#31333F")
    ax.spines["bottom"].set_color("#31333F")
    ax.spines["left"].set_color("#31333F")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")


if "show_more" not in st.session_state:
    st.session_state.show_more = False


# 조건을 설정한 후 확인하기를 눌러주세요 문구 추가
if not predict_button:
    st.markdown(
        "<p style='text-align: center; color: rgba(255, 255, 255, 0.5); font-size: 20px;'>좌측 사이드바에서 조건을 설정한 후 확인하기를 눌러주세요</p>",
        unsafe_allow_html=True,
    )

if predict_button:
    st.title(f"{name}님의 연체 예측 결과")

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

        st.subheader("개인 대출 정보")

        # 연수입
        user_income = selected_user["연간 소득"]
        income_percentile = 100 - (
            (total["AMT_INCOME_TOTAL"] < user_income).mean() * 100
        )

        st.markdown(
            "<p class='font-size-sub-subheader'>💶 나의 연수입</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='font-size: 16px;'>➔  내 연수입은 상위 <span style='color: #30e830; font-weight: bold;'>{income_percentile:.1f}%</span>에요.</p>",
            unsafe_allow_html=True,
        )

        # 부양 부담 지수 (Dependents_Index)
        dependents_index = selected_user.get("Dependents_Index", "정보 없음")

        st.markdown(
            "<p class='font-size-sub-subheader'>"
            f"👶 부양 부담 지수: <span style='color: #f6f6c5;'>{dependents_index}</span>"
            "</p>",
            unsafe_allow_html=True,
        )
        st.write(
            "➔  자녀에 대한 부양 부담이 가족 내에서 얼마나 큰 비중을 차지하는지를 나타내요."
        )

        # 소득 대비 부양 부담 지수 (Income_to_Dependents_Ratio)
        income_to_dependents_ratio = selected_user.get(
            "Income_to_Dependents_Ratio", "정보 없음"
        )

        st.markdown(
            "<p class='font-size-sub-subheader'>"
            f"🙋‍♂️🙋‍♀️ 소득 대비 부양 부담 지수: <span style='color: #f6f6c5;'>{income_to_dependents_ratio:.0f}</span>"
            "</p>",
            unsafe_allow_html=True,
        )
        st.write(
            "➔  개인의 소득이 자녀 부양에 얼마나 적절하게 분배될 수 있는지를 나타내요."
        )
        # 점선 추가
        st.markdown("<hr style='border: 1px dashed gray;' />", unsafe_allow_html=True)

        loan_count = int(selected_user["과거 대출 횟수"])  # 정수형으로 변환

        if loan_count > 0:
            st.subheader("과거 대출 이력")

            # 과거 대출 이력이 있는 경우 DSR 표현 추가
            if loan_count > 0:
                dsr = selected_user.get(
                    "부채 상환 가능성 지수", None
                )  # 'Debt_Repayment_Capability_Index' 컬럼 매핑된 이름 사용
            if dsr is not None:
                st.markdown(
                    "<p class='font-size-sub-subheader'>" f"💼 DSR" "</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='font-size: 16px;'><b>DSR이란?</b> '내 소득 중 빚 갚는 데 쓰는 돈의 비율'을 의미해요.</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='font-size: 16px;'>➔  내가 버는 총 소득 중에서 <span style='color: #30e830; font-weight: bold;'>{dsr:.2f}%</span>를 대출 상환에 쓰고 있어요.</p>",
                    unsafe_allow_html=True,
                )

            # Plotly Chart 1. 대출 대비 연체 횟수
            st.markdown(
                "<p class='font-size-sub-subheader'>" f"💸 대출 횟수" "</p>",
                unsafe_allow_html=True,
            )
            st.write(
                f"(대출 대비 연체 횟수 비율: {selected_user['대출 대비 연체 횟수 비율']})"
            )

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=total["LOAN_COUNT"],
                    name="",
                    hovertemplate="과거 대출 횟수: %{x}, Count: %{y}",
                    xbins=dict(
                        start=0, end=int(demo["과거 대출 횟수"].max()) + 1, size=1
                    ),
                    marker_color="#0064FF",
                )
            )
            fig.add_vline(
                x=loan_count,
                line_color="#FF4B4B",
                line_dash="dash",
                annotation_text=f"{name}: {loan_count}번",
                annotation_font_color="#FF4B4B",
            )
            fig.update_layout(
                title_text=f"전체 고객 과거 대출 횟수 분포 중 {name}님의 과거 대출 횟수",
                title_x=0.25,
                xaxis_title_text="과거 대출 횟수",
                bargap=0.05,
                # bargroupgap=0.1,
            )
            st.plotly_chart(figure_or_data=fig)

            # Plotly Chart 2. 대출 상환 비율 차트
            st.markdown(
                "<p class='font-size-sub-subheader'>" "💸 대출 상환 비율" "</p>",
                unsafe_allow_html=True,
            )

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=total["Credit_Utilization_Ratio"],
                    name="",
                    hovertemplate="대출 상환 비율: %{x}, Count: %{y}",
                    marker_color="#0064FF",
                )
            )
            fig.add_vline(
                x=selected_user["대출 상환 비율"],
                line_color="#FF4B4B",
                line_dash="dash",
                annotation_text=f"{name}: {selected_user['대출 상환 비율']:.2f}",
                annotation_font_color="#FF4B4B",
            )
            fig.update_xaxes(range=[0.0, 1.0])
            fig.update_yaxes(range=[0, 10000])
            fig.update_layout(
                title_text=f"전체 고객 대출 상환 비율 분포 중 {name}님의 비율",
                title_x=0.25,
                xaxis_title_text="대출 상환 비율",
                bargap=0.05,
            )
            st.plotly_chart(figure_or_data=fig)

            # Plotly Chart 3. 연 수입 대비 총 부채 비율 차트
            st.markdown(
                "<p class='font-size-sub-subheader'>"
                "💸 연 수입 대비 총 부채 비율"
                "</p>",
                unsafe_allow_html=True,
            )

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=total["Debt_to_Income_Ratio"],
                    name="",
                    hovertemplate="대출 상환 비율: %{x}, Count: %{y}",
                    marker_color="#0064FF",
                )
            )
            fig.add_vline(
                x=selected_user["부채 상환 비율"],
                line_color="#FF4B4B",
                line_dash="dash",
                annotation_text=f"{name}: {selected_user['부채 상환 비율']:.2f}",
                annotation_font_color="#FF4B4B",
            )
            fig.update_xaxes(range=[0.0, 5.0])
            fig.update_yaxes(range=[0, 13000])
            fig.update_layout(
                title_text=f"연 수입 대비 총 부채 비율 분포에서 {name}님의 비율",
                title_x=0.25,
                xaxis_title_text="연 수입 대비 총 부채 비율",
                bargap=0.05,
            )
            st.plotly_chart(figure_or_data=fig)

        else:
            st.subheader("과거 대출 이력")
            st.write("**조회할 대출 이력이 없습니다.**")

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

        # TEMP : API Call check
        body = (
            pd.DataFrame(selected_user)
            .transpose()
            .rename(columns=column_mapping_reverse)
        )
        eval_result = api.run_inference(
            df=body,
        )
        st.dataframe(eval_result)

        # 대출 가능성 평가 버튼
        evaluate_button = st.button("대출 가능성 평가")

        if evaluate_button:
            st.write("**대출 가능성 평가**")

            # 재직 기간 및 연수입 기준
            today = datetime.today()
            employment_duration_days = -selected_user["재직 여부"]
            employment_start_date = today - pd.to_timedelta(
                employment_duration_days, unit="D"
            )
            employment_duration_years = (today - employment_start_date).days / 365.25

            annual_income = selected_user["연간 소득"]

            # 대출 가능성 기준 설정
            min_employment_duration_years = 1  # 1년 이상
            min_annual_income = 20_000_000  # 연 2천만원 이상

            feedback = []

            if employment_duration_years < min_employment_duration_years:
                feedback.append("재직 기간이 부족합니다. 재직 기간을 늘려보세요.")
            else:
                feedback.append("재직 기간은 충분합니다.")

            if annual_income < min_annual_income:
                feedback.append("연수입이 부족합니다. 연수입을 늘려보세요.")
            else:
                feedback.append("연수입은 충분합니다.")

            if not feedback:  # feedback 리스트가 비어있다면
                feedback.append(
                    "대출 가능성이 높습니다. 추가적인 조건이 필요한 경우, 금융 기관에 문의하세요."
                )

            for line in feedback:
                st.write(f" - {line}")
