import os
import os.path as path
import platform
from datetime import datetime

# # 환경변수 설정
from dotenv import load_dotenv

load_dotenv(".env")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_cookies_controller import CookieController

controller = CookieController()
from utils import APIHelper, load_demo_data, load_total_data, download_data

# API Helper
api = APIHelper(
    api_url=os.environ.get("INFERENCE_API_URL"),
    api_key=os.environ.get("INFERENCE_API_KEY"),
)

# 데이터 로드
download_data()
demo = load_demo_data()
total = load_total_data()

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

    [data-testid='stSidebarHeader'] {
        display:none;
    }

    /* 페이지 링크 스타일 설정 */
    a[data-testid='stPageLink-NavLink'] {
    border: 2px solid rgba(255, 255, 255, 0.2) !important;  /* 연한 흰색 테두리 설정 */
    border-radius: 5px;  /* 테두리 둥글기 설정 */
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
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False

if "evaluate_clicked" not in st.session_state:
    st.session_state.evaluate_clicked = False

st.sidebar.image(
    image=Image.open(
        path.join(path.dirname(__file__), "../assets/TossBank_Logo_Primary.png")
    ),
    width=400,
    use_column_width=True,
)


st.sidebar.page_link(page="./app.py", label="Home", icon="🏠")

with st.sidebar.form(key="sidebar_form"):

    name = st.selectbox(
        "이름을 선택하세요",
        demo["이름"].unique(),
        index=None,
        placeholder=controller.get("selected_user_name"),
        disabled=True,
    )

    # 빈 공간 추가
    st.write(" ")
    st.write(" ")

    selected_loan_type = st.selectbox(
        "대출상품을 선택하세요",
        options=list(loan_types.keys()),
        format_func=lambda x: loan_types[x],
        index=None,
        placeholder=loan_types[controller.get("selected_loan_type")],
        disabled=True,
    )

    # 빈 공간 추가
    st.write(" ")
    st.write(" ")

    # text_input을 사용하여 대출 금액 직접 입력

    selected_amount = st.text_input(
        "대출 금액 입력 (예: 1, 2 ,..., 50)",
        value=int(controller.get("selected_amount_int") / 10000000),
        disabled=True,
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
    predict_button = st.form_submit_button("확인하기", disabled=True)
    if predict_button:
        st.session_state.predict_clicked = True
        st.session_state.evaluate_clicked = (
            False  # 새로운 확인하기 클릭 시 평가 상태 리셋
        )

# 본 화면

# '확인하기' 버튼이 눌렸는지 확인
if st.session_state.predict_clicked:
    name = controller.get("selected_user_name")
    st.title(f"{name}님의 대출 가능성 분석 결과")

    selected_user = demo[demo["이름"] == name]

    if selected_user.empty:
        st.write("선택된 사용자 데이터가 없습니다.")
    else:
        selected_user = selected_user.iloc[0]
        # 점선 추가
        st.markdown("<hr style='border: 1px dashed gray;' />", unsafe_allow_html=True)

        # TEMP : API Call check
        selected_user_df = (
            pd.DataFrame(selected_user)
            .transpose()
            .rename(columns=column_mapping_reverse)
        )
        selected_user_df["AMT_CREDIT"] = int(
            controller.get("selected_amount_int") / 1200
        )  # 환전 -> $
        selected_loan_type = controller.get("selected_loan_type")
        selected_user_df["NAME_CONTRACT_TYPE"] = (
            selected_loan_type
            if selected_loan_type == "Revolving loans"
            else "Cash loans"
        )
        # st.dataframe(selected_user_df)
        eval_result = api.run_inference(
            df=selected_user_df,
        ).iloc[0]
        eval_loan_proba = eval_result["pred_probs_loan"]
        eval_loan_approval = eval_result["preds_loan"]

        st.write("")

        st.markdown(
            f"<p style='font-size: 28px; font-weight: bold;'>대출 승인 확률 :  <span style='color: #80fdc3; font-weight: bold;'>{(eval_loan_proba*100):.2f}%</p>",
            unsafe_allow_html=True,
        )

        # 승인 여부에 따라 색상을 지정합니다.
        approval_text = "승인" if eval_loan_approval else "미승인"
        approval_color = "#80fdc3" if eval_loan_approval else "#fee34d"

        # st.markdown을 사용하여 HTML 마크업을 출력합니다.
        st.markdown(
            f"<p style='font-size: 28px; font-weight: bold;'>신청하신 대출은 <span style='color: {approval_color}; font-weight: bold;'> {approval_text} </span> 될 가능성이 높아요.</p>",
            unsafe_allow_html=True,
        )

        st.write("")

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

            # 빈 공간 추가
            st.write(" ")
            st.write(" ")
            # Plotly Chart 1. 대출 대비 연체 횟수
            st.markdown(
                f"""
                <p class='font-size-sub-subheader'>
                    💸 대출 횟수 <br>
                    <span style='color: rgba(255, 255, 255, 0.5); font-size: 0.8em;'>
                        ({name}님의 과거 대출 횟수: 
                        <span style='color: rgba(255, 75, 75, 0.7);'>{loan_count}번</span>)
                    </span>
                </p>
                """,
                unsafe_allow_html=True,
            )

            st.write(" ")
            st.markdown(
                f"""
                <p class='font-size-sub-subheader' align='center'>
                    <span style='font-size: 0.8em;'>
                        전체 고객 과거 대출 횟수 분포 중 {name}님의 과거 대출 횟수
                    </span>
                </p>
                """,
                unsafe_allow_html=True,
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
                # title_text=f"전체 고객 과거 대출 횟수 분포 중 {name}님의 과거 대출 횟수",
                # title_x=0.25,
                margin=dict(
                    t=20,
                ),
                xaxis_title_text="과거 대출 횟수",
                bargap=0.05,
                # bargroupgap=0.1,
            )
            st.plotly_chart(figure_or_data=fig)

            # 빈 공간 추가
            st.write(" ")
            # Plotly Chart 2. 대출 상환 비율 차트
            st.markdown(
                "<p class='font-size-sub-subheader'>" "💸 잔여 부채 비율" "</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='font-size: 16px;'><b>잔여 부채 비율이란?</b> 전체 대출금 중 부채가 얼마나 남았는지 보여주는 비율이에요.</p>",
                unsafe_allow_html=True,
            )
            percent_of_CUR = selected_user["대출 상환 비율"] * 100
            st.markdown(
                f"<p style='font-size: 16px;'>➔  {name}님의 부채는 <span style='color: #30e830; font-weight: bold;'> 약 {percent_of_CUR:.0f}%</span>가 남았어요.</p>",
                unsafe_allow_html=True,
            )

            st.write(" ")
            st.markdown(
                f"""
                <p class='font-size-sub-subheader' align='center'>
                    <span style='font-size: 0.8em;'>
                        전체 고객 대출 상환 비율 분포 중 {name}님의 비율
                    </span>
                </p>
                """,
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
                # title_text=f"전체 고객 대출 상환 비율 분포 중 {name}님의 비율",
                # title_x=0.25,
                margin=dict(
                    t=20,
                ),
                xaxis_title_text="대출 상환 비율",
                bargap=0.05,
            )
            st.plotly_chart(figure_or_data=fig)

            # 빈 공간 추가
            st.write(" ")
            # DSR
            dsr = selected_user.get("부채 상환 가능성 지수", None)
            # 'Debt_Repayment_Capability_Index' 컬럼 매핑된 이름 사용

            percent_of_DSR = 100 * dsr

            if dsr is not None:
                st.markdown(
                    "<p class='font-size-sub-subheader'>" f"💼 DSR" "</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='font-size: 16px;'><b>DSR이란?</b> 내 소득 중 빚 갚는 데 쓰는 돈의 비율이에요.</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='font-size: 16px;'>➔  {name}님은 총 소득 중에서 <span style='color: #30e830; font-weight: bold;'>약 {percent_of_DSR:.0f}%</span>를 대출 상환에 쓰고 있어요.</p>",
                    unsafe_allow_html=True,
                )

            # 빈 공간 추가
            st.write(" ")
            st.write(" ")  # 빈 공간 추가
            st.write(" ")
            st.write(" ")
            # Plotly Chart 3. 연 수입 대비 총 부채 비율 차트
            st.markdown(
                "<p class='font-size-sub-subheader'>"
                "💸 연 수입 대비 총 부채 비율"
                "</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='font-size: 16px;'><b>연 수입 대비 총 부채 비율이란?</b> 1년 동안 버는 돈에 비해 얼마나 많은 부채를 가지고 있는지를 보여주는 비율이에요.</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='font-size: 16px;'>➔  {name}님의 부채는 연수입의 <span style='color: #30e830; font-weight: bold;'> 약 {selected_user['부채 상환 비율']:.2f}배</span>에요.</p>",
                unsafe_allow_html=True,
            )

            st.write(" ")
            st.markdown(
                f"""
                <p class='font-size-sub-subheader' align='center'>
                    <span style='font-size: 0.8em;'>
                        연 수입 대비 총 부채 비율 분포에서 {name}님의 비율
                    </span>
                </p>
                """,
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
                # title_text=f"연 수입 대비 총 부채 비율 분포에서 {name}님의 비율",
                # title_x=0.25,
                margin=dict(
                    t=20,
                ),
                xaxis_title_text="연 수입 대비 총 부채 비율",
                bargap=0.05,
            )
            st.plotly_chart(figure_or_data=fig)

        else:
            st.subheader("과거 대출 이력")
            st.write("**조회할 대출 이력이 없습니다.**")
