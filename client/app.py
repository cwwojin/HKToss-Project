import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
from datetime import datetime
import os
import platform

# # 데이터셋 경로 설정
# data_path = os.path.join("data", "demo_set.csv")


# 추후 경로 다시 설정할게요!!
data_path = "/Users/khb43/Desktop/HANKYUNG_WITH_TOSS_BANK-2(소민호 강사님)/week9~11 실무 프로젝트 관련/data/선정 데이터(프로젝트 데이터)/clients/demo_set.csv"


# font_name = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font_name)

# OS 별 폰트 깨짐 처리
if platform.system() == "Darwin":  # 맥
    plt.rc("font", family="AppleGothic")
elif platform.system() == "Windows":  # 윈도우
    plt.rc("font", family="Malgun Gothic")
elif platform.system() == "Linux":  # 리눅스 (구글 콜랩)
    #!wget "https://www.wfonts.com/download/data/2016/06/13/malgun-gothic/malgun.ttf"
    #!mv malgun.ttf /usr/share/fonts/truetype/
    # import matplotlib.font_manager as fm
    # fm._rebuild()
    plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False  # 한글 폰트 사용시 마이너스 폰트 깨짐 해결


@st.cache_data
def load_data():
    df = pd.read_csv(data_path, low_memory=False)
    return df


# 데이터 로드
demo = load_data()

# 열 이름 매핑
column_mapping = {
    "Credit_Utilization_Ratio": "신용 사용 비율",
    "Debt_to_Income_Ratio": "부채 상환 비율",
    "OVERDUE_RATIO": "💸 대출 대비 연체 횟수",
    "Debt_Repayment_Capability_Index": "부채 상환 가능성 지수",
    "LOAN_COUNT": "과거 대출 횟수",
    "AMT_CREDIT": "현재 대출 금액",
    "name": "이름",
    "DAYS_BIRTH": "😀 나이",
    "FLAG_MOBIL": "📱 휴대전화 소유 여부",
    "FLAG_OWN_CAR": "🚗 자차 소유 여부",  # 열 이름 확인 후 올바르게 수정
    "FLAG_OWN_REALTY": "🏡 부동산 소유 여부",
    "DAYS_EMPLOYED": "🏢 재직 여부",
    "AMT_INCOME_TOTAL": "연간 소득",
    "LOAN_STATUS": "대출 상태",
}

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

# 사이드바 위젯 구성
name = st.sidebar.selectbox("이름을 선택하세요", demo["이름"].unique())

# 빈 공간 추가
st.sidebar.write(" ")
st.sidebar.write(" ")

selected_loan_type = st.sidebar.selectbox(
    "대출상품을 선택하세요",
    options=list(loan_types.keys()),
    format_func=lambda x: loan_types[x],
)

# 빈 공간 추가
st.sidebar.write(" ")
st.sidebar.write(" ")

credit_min = st.sidebar.slider(
    "대출 금액 범위 선택 (최대값: ₩50,000,000은 선택 불가)",
    min_value=1_000_000,
    max_value=50_000_000,
    value=1_000_000,
    step=1_000_000,
    format="₩%d",
)

# 최댓값은 50,000,000으로 고정
credit_max = 50_000_000
credit_range_text = (
    f"₩{credit_min // 1_000_000}천만 원 ~ ₩{credit_max // 1_000_000}천만 원"
)
st.sidebar.write(f"선택된 대출 금액 범위: {credit_range_text}")

# '확인하기' 버튼을 추가하여 연체 예측 결과를 확인
predict_button = st.sidebar.button("확인하기")

# 데이터 필터링
filtered_demo = demo[
    (demo["현재 대출 금액"] >= credit_min) & (demo["현재 대출 금액"] <= credit_max)
]


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

if predict_button:
    st.header(f"{name}님의 연체 예측 결과")

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

        age = calculate_age(selected_user["😀 나이"])

        st.subheader("현재 나의 정보")
        st.markdown(f"**😀 나이:** {age} 세")
        st.markdown(
            f"**📱 휴대전화 소유 여부:** {'Y' if selected_user['📱 휴대전화 소유 여부'] else 'N'}"
        )
        st.markdown(
            f"**🚗 자차 소유 여부:** {'Y' if selected_user['🚗 자차 소유 여부'] else 'N'}"
        )
        st.markdown(
            f"**🏡 부동산 소유 여부:** {'Y' if selected_user['🏡 부동산 소유 여부'] else 'N'}"
        )
        st.markdown(
            f"**🏢 재직 여부:** {'Y' if selected_user['🏢 재직 여부'] else 'N'}"
        )

        # 점선 추가
        st.markdown("<hr style='border: 1px dashed #000;' />", unsafe_allow_html=True)

        # 대출 대비 연체 횟수
        st.write(
            f"**💸 대출 대비 연체 횟수:** {selected_user['💸 대출 대비 연체 횟수']}"
        )

        # 대출 이력 연체 횟수 차트
        loan_count = int(selected_user["과거 대출 횟수"])  # 정수형으로 변환
        if loan_count > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            create_style(ax)
            ax.set_title("과거 대출 횟수 분포", color="white")

            bins_range = range(0, int(demo["과거 대출 횟수"].max()) + 1)
            sns.histplot(
                demo["과거 대출 횟수"],
                kde=False,
                ax=ax,
                color="lightblue",
                bins=bins_range,
            )

            # 사용자 포인트 표시
            ax.axvline(loan_count, color="red", linestyle="--")
            ax.text(
                loan_count,
                ax.get_ylim()[1] * 0.9,
                f"{name}: {loan_count}번",
                color="#FF4B4B",
                ha="center",
            )

            st.pyplot(fig)

            # 대출 이력 연체 횟수 차트
            overdue_data = demo[demo["💸 대출 대비 연체 횟수"] > 0]
            if not overdue_data.empty:
                st.write("**대출 대비 연체 횟수 분포**")
                fig, ax = plt.subplots(figsize=(8, 4))
                create_style(ax)
                ax.set_title("대출 대비 연체 횟수 분포", color="white")

                sns.countplot(
                    x="💸 대출 대비 연체 횟수",
                    data=overdue_data,
                    ax=ax,
                    palette="pastel",
                )
                st.pyplot(fig)

            # 신용 사용 비율 차트
            st.write("신용 사용 비율")
            fig, ax = plt.subplots(figsize=(8, 4))
            create_style(ax)
            ax.set_title("신용 사용 비율 분포", color="white")

            sns.histplot(demo["신용 사용 비율"], kde=True, ax=ax, color="skyblue")

            # 사용자 포인트 표시
            ax.axvline(selected_user["신용 사용 비율"], color="red", linestyle="--")
            ax.text(
                selected_user["신용 사용 비율"],
                ax.get_ylim()[1] * 0.9,
                f'{name}: {selected_user["신용 사용 비율"]:.2f}',
                color="#FF4B4B",
                ha="center",
            )

            st.pyplot(fig)

            # 연 수입 대비 총 부채 비율 차트
            st.write("연 수입 대비 총 부채 비율")
            fig, ax = plt.subplots(figsize=(8, 4))
            create_style(ax)
            ax.set_title("연 수입 대비 총 부채 비율 분포", color="white")

            sns.histplot(demo["부채 상환 비율"], kde=True, ax=ax, color="salmon")

            # 사용자 포인트 표시
            ax.axvline(selected_user["부채 상환 비율"], color="red", linestyle="--")
            ax.text(
                selected_user["부채 상환 비율"],
                ax.get_ylim()[1] * 0.9,
                f'{name}: {selected_user["부채 상환 비율"]:.2f}',
                color="#FF4B4B",
                ha="center",
            )

            st.pyplot(fig)

        else:
            st.write("**조회할 대출 이력이 없습니다.**")

        # 점선 추가
        st.markdown("<hr style='border: 1px dashed #000;' />", unsafe_allow_html=True)

        # 대출 상품 정보와 평가 버튼 추가
        st.subheader("신청한 대출 정보")
        st.write(f"선택한 대출 상품: {selected_loan_type}")

        # 대출 가능성 평가 버튼
        evaluate_button = st.button("대출 가능성 평가")

        if evaluate_button:
            st.write("**대출 가능성 평가**")

            # 재직 기간 및 연수입 기준
            today = datetime.today()
            employment_duration_days = -selected_user["🏢 재직 여부"]
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
