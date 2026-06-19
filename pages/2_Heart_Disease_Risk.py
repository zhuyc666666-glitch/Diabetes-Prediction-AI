from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# 心脏病最佳模型路径：不要修改，页面从这里加载 train_heart.py 生成的模型。
MODEL_PATH = Path("models") / "best_heart_model.pkl"

# 心脏病训练特征列路径：不要修改，用于保证页面输入和训练时列名一致。
FEATURE_COLUMNS_PATH = Path("models") / "heart_feature_columns.pkl"

# 心脏病特征重要性路径：用于展示 Top 5 Risk Factors。
FEATURE_IMPORTANCE_PATH = Path("models") / "heart_feature_importance.csv"


def configure_page():
    """配置心脏病风险预测页面和自定义 CSS。"""
    st.set_page_config(
        page_title="Heart Disease Risk Prediction | MedPredict AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 使用和糖尿病页面一致的 Apple 医疗科技风：
    # 白色背景、浅蓝绿色渐变、圆角卡片、轻微阴影和大面积留白。
    st.markdown(
        """
        <style>
        :root {
            --bg: #f7fbfc;
            --ink: #102a43;
            --muted: #5f7384;
            --line: rgba(14, 116, 144, 0.14);
            --teal: #0e7490;
            --green: #16a34a;
            --orange: #f59e0b;
            --red: #dc2626;
            --shadow: 0 24px 70px rgba(15, 61, 74, 0.10);
        }

        html, body, .stApp {
            background:
                radial-gradient(circle at top left, rgba(125, 211, 252, 0.26), transparent 34rem),
                radial-gradient(circle at top right, rgba(94, 234, 212, 0.22), transparent 32rem),
                linear-gradient(180deg, #ffffff 0%, var(--bg) 100%);
            color: var(--ink);
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
        }

        .main .block-container {
            max-width: 1320px;
            padding-top: 2.4rem;
            padding-bottom: 3rem;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0);
        }

        h1, h2, h3, p {
            letter-spacing: 0;
        }

        .hero {
            padding: 3.2rem 1rem 2.3rem;
            text-align: center;
        }

        .hero-pill {
            display: inline-flex;
            padding: 0.45rem 0.85rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid var(--line);
            color: var(--teal);
            font-size: 0.86rem;
            font-weight: 650;
            box-shadow: 0 10px 28px rgba(15, 61, 74, 0.06);
        }

        .hero h1 {
            margin: 1rem 0 0.55rem;
            font-size: clamp(2.8rem, 6vw, 5.4rem);
            line-height: 0.98;
            font-weight: 790;
            color: #082f3b;
        }

        .hero p {
            max-width: 760px;
            margin: 0 auto;
            color: var(--muted);
            font-size: 1.06rem;
            line-height: 1.7;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid rgba(14, 116, 144, 0.13);
            border-radius: 28px;
            box-shadow: var(--shadow);
            padding: 1.6rem;
            backdrop-filter: blur(18px);
        }

        div[data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid rgba(14, 116, 144, 0.13);
            border-radius: 28px;
            box-shadow: var(--shadow);
            padding: 1.6rem;
            backdrop-filter: blur(18px);
        }

        .card-kicker {
            color: var(--teal);
            font-size: 0.8rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }

        .card-title {
            margin: 0;
            color: #0f3d4c;
            font-size: 1.55rem;
            line-height: 1.2;
            font-weight: 760;
        }

        .card-subtitle {
            margin: 0.45rem 0 1.1rem;
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        div[data-testid="stSelectbox"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stTextInput"] label {
            color: #294b5a;
            font-weight: 640;
            font-size: 0.9rem;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            border-radius: 14px;
            border-color: rgba(14, 116, 144, 0.18);
            background: rgba(255, 255, 255, 0.94);
        }

        div.stButton > button {
            width: 100%;
            border: 0;
            border-radius: 18px;
            padding: 0.86rem 1rem;
            color: white;
            background: linear-gradient(135deg, #0e7490 0%, #14b8a6 100%);
            font-weight: 760;
            box-shadow: 0 18px 35px rgba(14, 116, 144, 0.22);
        }

        div.stButton > button:hover {
            color: white;
            transform: translateY(-1px);
            box-shadow: 0 22px 45px rgba(14, 116, 144, 0.27);
        }

        .result-panel {
            min-height: 455px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            gap: 1rem;
        }

        .risk-ring {
            width: 210px;
            height: 210px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background:
                radial-gradient(circle at center, #ffffff 0 58%, transparent 59%),
                conic-gradient(var(--risk-color) calc(var(--risk-value) * 1%), #e8f1f3 0);
            box-shadow: inset 0 0 0 1px rgba(14, 116, 144, 0.10), 0 18px 45px rgba(15, 61, 74, 0.10);
        }

        .risk-number {
            font-size: 2.65rem;
            font-weight: 800;
            color: var(--risk-color);
        }

        .risk-label {
            padding: 0.52rem 0.9rem;
            border-radius: 999px;
            color: var(--risk-color);
            background: var(--risk-bg);
            font-weight: 760;
            font-size: 1rem;
        }

        .risk-copy {
            max-width: 350px;
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.55;
        }

        .section-title {
            margin: 2rem 0 0.8rem;
            font-size: 1.35rem;
            font-weight: 720;
            color: #0f3d4c;
        }

        .factor-row {
            margin: 0.9rem 0;
        }

        .factor-head {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.35rem;
            color: #294b5a;
            font-size: 0.94rem;
            font-weight: 650;
        }

        .bar-track {
            height: 12px;
            border-radius: 999px;
            background: #e7f2f4;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #0e7490 0%, #5eead4 100%);
        }

        .soft-note {
            color: var(--muted);
            background: rgba(255, 255, 255, 0.74);
            border: 1px dashed rgba(14, 116, 144, 0.25);
            border-radius: 18px;
            padding: 1rem;
        }

        @media (max-width: 900px) {
            .hero {
                padding-top: 2rem;
            }
            .risk-ring {
                width: 180px;
                height: 180px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model_and_feature_columns():
    """加载心脏病最佳模型和训练时保存的原始特征列。"""
    # 如果还没有运行 train_heart.py，页面只显示提示，不抛出异常。
    if not MODEL_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
        return None, None

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, feature_columns


@st.cache_data
def load_feature_importance():
    """读取心脏病模型特征重要性 CSV。"""
    if not FEATURE_IMPORTANCE_PATH.exists():
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    if {"feature", "importance"}.issubset(importance_df.columns):
        return importance_df.sort_values("importance", ascending=False)

    return pd.DataFrame(columns=["feature", "importance"])


def render_hero():
    """渲染页面顶部 Hero 区域。"""
    st.markdown(
        """
        <section class="hero">
            <div class="hero-pill">Cardiovascular AI Prototype</div>
            <h1>Heart Disease Risk Prediction</h1>
            <p>
                A machine learning module for educational heart disease risk assessment
                using structured clinical indicators.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_missing_model_message():
    """模型文件缺失时显示友好提示。"""
    st.markdown(
        """
        <div class="glass-card">
            <div class="card-kicker">Model Required</div>
            <h3 class="card-title">Please run train_heart.py first.</h3>
            <p class="card-subtitle">
                The heart disease model and feature columns are required before prediction can run.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_input(feature_name):
    """根据特征列名称自动生成合适的输入控件。"""
    # 常见字段使用更符合医学表单习惯的控件，同时输出值保持训练时的数值格式。
    if feature_name == "age":
        return st.number_input("age", min_value=0.0, max_value=120.0, value=55.0, step=1.0)
    if feature_name == "sex":
        return st.selectbox(
            "sex",
            [1.0, 0.0],
            format_func=lambda value: "Male" if value == 1.0 else "Female",
        )
    if feature_name == "cp":
        return st.selectbox("cp", [1.0, 2.0, 3.0, 4.0])
    if feature_name == "trestbps":
        return st.number_input("trestbps", min_value=60.0, max_value=250.0, value=130.0, step=1.0)
    if feature_name == "chol":
        return st.number_input("chol", min_value=80.0, max_value=700.0, value=240.0, step=1.0)
    if feature_name == "fbs":
        return st.selectbox("fbs", [0.0, 1.0])
    if feature_name == "restecg":
        return st.selectbox("restecg", [0.0, 1.0, 2.0])
    if feature_name == "thalach":
        return st.number_input("thalach", min_value=50.0, max_value=250.0, value=150.0, step=1.0)
    if feature_name == "exang":
        return st.selectbox("exang", [0.0, 1.0])
    if feature_name == "oldpeak":
        return st.number_input("oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    if feature_name == "slope":
        return st.selectbox("slope", [0.0, 1.0, 2.0])
    if feature_name == "ca":
        return st.selectbox("ca", [0.0, 1.0, 2.0, 3.0, 4.0])
    if feature_name == "thal":
        return st.selectbox("thal", [0.0, 1.0, 2.0, 3.0])
    if feature_name == "source":
        return st.selectbox("source", ["cleveland", "hungarian", "switzerland", "va"])

    # 未知特征默认使用数字输入，保持页面能随 feature_columns 自动扩展。
    return st.number_input(feature_name, value=0.0, step=1.0)


def render_patient_card(feature_columns):
    """渲染左侧输入卡片，并按 feature_columns 收集输入数据。"""
    with st.form("heart_prediction_form"):
        st.markdown(
            """
            <div class="card-kicker">Patient Profile</div>
            <h3 class="card-title">Clinical Information</h3>
            <p class="card-subtitle">Enter cardiovascular indicators for a single risk estimate.</p>
            """,
            unsafe_allow_html=True,
        )

        input_data = {}

        # 按训练时保存的原始特征列自动生成控件，确保输入列完整且顺序可对齐。
        for index in range(0, len(feature_columns), 2):
            cols = st.columns(2)
            for offset, column in enumerate(cols):
                feature_index = index + offset
                if feature_index >= len(feature_columns):
                    continue

                feature_name = feature_columns[feature_index]
                with column:
                    input_data[feature_name] = render_feature_input(feature_name)

        submitted = st.form_submit_button("Predict Heart Disease Risk")

    return submitted, input_data


def align_features(input_data, feature_columns):
    """把用户输入整理成和训练时原始特征列完全一致的 DataFrame。"""
    input_df = pd.DataFrame([input_data])
    return input_df.reindex(columns=feature_columns)


def predict_probability(model, input_df):
    """使用模型预测 heart disease=1 的概率。"""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(input_df)[0, 1])

    st.error("The loaded heart disease model does not support probability prediction.")
    st.stop()


def classify_risk(probability):
    """根据概率划分风险等级。"""
    if probability < 0.3:
        return "Low Risk"
    if probability < 0.7:
        return "Medium Risk"
    return "High Risk"


def get_risk_style(risk_level):
    """根据风险等级返回颜色配置。"""
    if risk_level == "Low Risk":
        return "#16a34a", "rgba(22, 163, 74, 0.12)"
    if risk_level == "Medium Risk":
        return "#f59e0b", "rgba(245, 158, 11, 0.14)"
    return "#dc2626", "rgba(220, 38, 38, 0.12)"


def render_result_card(probability=None, risk_level=None):
    """渲染右侧预测结果卡片。"""
    if probability is None or risk_level is None:
        probability = 0.0
        risk_level = "Awaiting Assessment"
        risk_color = "#0e7490"
        risk_bg = "rgba(14, 116, 144, 0.10)"
        summary = "Complete the clinical profile and run the assessment to view predicted risk."
    else:
        risk_color, risk_bg = get_risk_style(risk_level)
        summary = "This probability is generated by the trained heart disease model."

    risk_percent = probability * 100
    st.markdown(
        f"""
        <div class="glass-card result-panel">
            <div>
                <div class="card-kicker">Risk Assessment</div>
                <h3 class="card-title">Prediction Result</h3>
            </div>
            <div class="risk-ring" style="--risk-color: {risk_color}; --risk-value: {risk_percent:.2f};">
                <div class="risk-number">{risk_percent:.1f}%</div>
            </div>
            <div class="risk-label" style="--risk-color: {risk_color}; --risk-bg: {risk_bg};">
                {risk_level}
            </div>
            <div class="risk-copy">
                <strong>Heart Disease Risk Probability</strong><br>
                {summary}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_factor_name(feature_name):
    """把模型特征名转换成更适合展示的英文标签。"""
    return feature_name.replace("_", " ").title()


def render_top_risk_factors():
    """展示 Top 5 Risk Factors。"""
    importance_df = load_feature_importance()

    st.markdown('<div class="section-title">Model Explanation</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
            <div class="card-kicker">Interpretability</div>
            <h3 class="card-title">Top 5 Risk Factors</h3>
            <p class="card-subtitle">Feature importance from the trained heart disease model.</p>
        """,
        unsafe_allow_html=True,
    )

    if importance_df.empty:
        st.markdown(
            '<div class="soft-note">Feature importance is not available yet. Please run train_heart.py first.</div>',
            unsafe_allow_html=True,
        )
    else:
        top_factors = importance_df.head(5).copy()
        max_importance = max(float(top_factors["importance"].max()), 1e-9)

        for _, row in top_factors.iterrows():
            feature = format_factor_name(str(row["feature"]))
            importance = float(row["importance"])
            width = min(max(importance / max_importance * 100, 2), 100)
            st.markdown(
                f"""
                <div class="factor-row">
                    <div class="factor-head">
                        <span>{feature}</span>
                        <span>{importance:.4f}</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: {width:.1f}%"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    """心脏病风险预测页面入口。"""
    configure_page()
    render_hero()

    model, feature_columns = load_model_and_feature_columns()
    if model is None or feature_columns is None:
        render_missing_model_message()
        return

    # 使用 session_state 保存最近一次预测结果。
    if "heart_probability" not in st.session_state:
        st.session_state.heart_probability = None
    if "heart_risk_level" not in st.session_state:
        st.session_state.heart_risk_level = None

    left_col, right_col = st.columns([1.08, 0.92], gap="large")
    with left_col:
        submitted, input_data = render_patient_card(feature_columns)

    if submitted:
        input_df = align_features(input_data, feature_columns)
        probability = predict_probability(model, input_df)
        st.session_state.heart_probability = probability
        st.session_state.heart_risk_level = classify_risk(probability)

    with right_col:
        render_result_card(
            st.session_state.heart_probability,
            st.session_state.heart_risk_level,
        )

    render_top_risk_factors()


if __name__ == "__main__":
    main()
