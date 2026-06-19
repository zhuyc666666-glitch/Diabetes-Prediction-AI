from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# 模型文件路径：不要修改，部署时 app.py 会从这里加载训练好的最佳模型
MODEL_PATH = Path("models") / "best_diabetes_model.pkl"

# 特征列文件路径：不要修改，用于保证网页输入和训练时特征列对齐
FEATURE_COLUMNS_PATH = Path("models") / "feature_columns.pkl"

# 特征重要性文件路径：由 train.py 生成，用于展示 Top 5 Risk Factors
FEATURE_IMPORTANCE_PATH = Path("models") / "feature_importance.csv"

# 模型表现图片路径：如果图片不存在，页面只显示提示文字，不报错
CONFUSION_MATRIX_PATH = Path("figures") / "confusion_matrix.png"
ROC_CURVE_PATH = Path("figures") / "roc_curve.png"


def configure_page():
    """配置 Streamlit 页面和自定义 CSS。"""
    st.set_page_config(
        page_title="MedPredict AI",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # 使用 CSS 构建 Apple 风格的医疗 AI 产品界面：
    # 白色背景、浅蓝绿色渐变、大留白、圆角卡片和轻微阴影。
    st.markdown(
        """
        <style>
        :root {
            --bg: #f7fbfc;
            --ink: #102a43;
            --muted: #5f7384;
            --line: rgba(14, 116, 144, 0.14);
            --teal: #0e7490;
            --teal-soft: #e6f7f8;
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

        h1, h2, h3, p {
            letter-spacing: 0;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0);
        }

        .hero {
            padding: 3.4rem 1rem 2.5rem;
            text-align: center;
        }

        .hero-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 0.85rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid var(--line);
            color: var(--teal);
            font-size: 0.86rem;
            font-weight: 650;
            box-shadow: 0 10px 28px rgba(15, 61, 74, 0.06);
        }

        .hero h1 {
            margin: 1rem 0 0.4rem;
            font-size: clamp(3rem, 6vw, 5.7rem);
            line-height: 0.96;
            font-weight: 780;
            color: #082f3b;
        }

        .hero h2 {
            margin: 0.3rem 0 0.7rem;
            font-size: clamp(1.35rem, 2.5vw, 2.15rem);
            line-height: 1.15;
            font-weight: 620;
            color: #0e7490;
        }

        .hero p {
            max-width: 720px;
            margin: 0 auto;
            color: var(--muted);
            font-size: 1.08rem;
            line-height: 1.7;
        }

        .section-title {
            margin: 2rem 0 0.8rem;
            font-size: 1.35rem;
            font-weight: 720;
            color: #0f3d4c;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.82);
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

        div[data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(14, 116, 144, 0.13);
            border-radius: 28px;
            box-shadow: var(--shadow);
            padding: 1.6rem;
            backdrop-filter: blur(18px);
        }

        div[data-testid="stSelectbox"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSlider"] label {
            color: #294b5a;
            font-weight: 640;
            font-size: 0.9rem;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {
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
            max-width: 330px;
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.55;
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

        .performance-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1.2rem;
        }

        .soft-note {
            color: var(--muted);
            background: rgba(255, 255, 255, 0.74);
            border: 1px dashed rgba(14, 116, 144, 0.25);
            border-radius: 18px;
            padding: 1rem;
        }

        .disclaimer {
            margin: 2.3rem 0 0;
            padding: 1.1rem 1.25rem;
            border-radius: 22px;
            text-align: center;
            color: #516778;
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid rgba(14, 116, 144, 0.12);
        }

        @media (max-width: 900px) {
            .hero {
                padding-top: 2rem;
            }
            .performance-grid {
                grid-template-columns: 1fr;
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
    """加载训练好的模型和特征列。"""
    # 如果模型文件不存在，页面停止运行并提示用户先训练模型
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.info("Please run train.py first to generate the trained model.")
        st.stop()

    # 如果特征列文件不存在，页面无法保证输入和训练特征一致
    if not FEATURE_COLUMNS_PATH.exists():
        st.error(f"Feature columns file not found: {FEATURE_COLUMNS_PATH}")
        st.info("Please run train.py first to generate feature_columns.pkl.")
        st.stop()

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, feature_columns


@st.cache_data
def load_feature_importance():
    """读取特征重要性 CSV。"""
    # 如果文件不存在，返回空表，页面会显示提示文字
    if not FEATURE_IMPORTANCE_PATH.exists():
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    if {"feature", "importance"}.issubset(importance_df.columns):
        return importance_df.sort_values("importance", ascending=False)

    return pd.DataFrame(columns=["feature", "importance"])


def render_hero():
    """渲染顶部 Hero 区域。"""
    st.markdown(
        """
        <section class="hero">
            <div class="hero-pill">Clinical AI Prototype</div>
            <h1>MedPredict AI</h1>
            <h2>AI-powered Diabetes Risk Assessment</h2>
            <p>
                An interpretable machine learning tool for educational diabetes risk prediction.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_patient_card():
    """渲染左侧 Patient Information 输入卡片。"""
    # 使用表单保证所有输入一次性提交，避免用户调整一个字段就触发预测
    with st.form("diabetes_prediction_form"):
        st.markdown(
            """
            <div class="card-kicker">Patient Profile</div>
            <h3 class="card-title">Patient Information</h3>
            <p class="card-subtitle">Enter clinical indicators and lifestyle information for a single risk estimate.</p>
            """,
            unsafe_allow_html=True,
        )

        gender = st.selectbox("gender", ["Female", "Male", "Other"])
        age = st.slider("age", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
        hypertension = st.selectbox(
            "hypertension",
            [0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )
        heart_disease = st.selectbox(
            "heart_disease",
            [0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )
        smoking_history = st.selectbox(
            "smoking_history",
            ["never", "No Info", "current", "former", "ever", "not current"],
        )

        metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
        with metric_col_1:
            bmi = st.number_input("bmi", min_value=10.0, max_value=80.0, value=27.5, step=0.1)
        with metric_col_2:
            hba1c_level = st.number_input(
                "HbA1c_level",
                min_value=3.0,
                max_value=15.0,
                value=5.8,
                step=0.1,
            )
        with metric_col_3:
            blood_glucose_level = st.number_input(
                "blood_glucose_level",
                min_value=50,
                max_value=400,
                value=140,
                step=1,
            )

        submitted = st.form_submit_button("Predict Diabetes Risk")

    # 字段名必须保持和训练数据一致，后续才能和 feature_columns 对齐
    input_data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c_level,
        "blood_glucose_level": blood_glucose_level,
    }

    return submitted, input_data


def align_features(input_data, feature_columns):
    """将用户输入处理成和训练时特征列一致的 DataFrame。"""
    raw_df = pd.DataFrame([input_data])

    # 当前训练脚本保存的是原始特征列，Pipeline 内部负责 One-Hot Encoding
    if set(feature_columns).issubset(set(raw_df.columns)):
        return raw_df.reindex(columns=feature_columns)

    # 如果后续改为保存 One-Hot 后的列，这里也能自动展开并补齐缺失列
    encoded_df = pd.get_dummies(
        raw_df,
        columns=["gender", "smoking_history"],
        dtype=int,
    )
    return encoded_df.reindex(columns=feature_columns, fill_value=0)


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


def predict_probability(model, input_df):
    """使用模型预测 diabetes=1 的概率。"""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(input_df)[0, 1])

    st.error("The loaded model does not support probability prediction.")
    st.stop()


def render_result_card(probability=None, risk_level=None):
    """渲染右侧 Prediction Result 卡片。"""
    if probability is None or risk_level is None:
        probability = 0.0
        risk_level = "Awaiting Assessment"
        risk_color = "#0e7490"
        risk_bg = "rgba(14, 116, 144, 0.10)"
        summary = "Complete the patient profile and run the assessment to view the predicted risk."
    else:
        risk_color, risk_bg = get_risk_style(risk_level)
        summary = "This probability is generated by the trained machine learning pipeline."

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
            <div class="risk-copy">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_factor_name(feature_name):
    """把模型特征名转换成更适合展示的英文标签。"""
    return feature_name.replace("_", " ").replace("HbA1c", "HbA1c").title()


def render_model_explanation():
    """渲染 Model Explanation 区域。"""
    importance_df = load_feature_importance()

    st.markdown('<div class="section-title">Model Explanation</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
            <div class="card-kicker">Interpretability</div>
            <h3 class="card-title">Top 5 Risk Factors</h3>
            <p class="card-subtitle">Feature importance from the selected tree-based model.</p>
        """,
        unsafe_allow_html=True,
    )

    if importance_df.empty:
        st.markdown(
            '<div class="soft-note">Feature importance is not available yet. Please run train.py first.</div>',
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


def render_model_performance():
    """渲染 Model Performance 区域。"""
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="performance-grid">', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-kicker">Evaluation</div><h3 class="card-title">Confusion Matrix</h3>', unsafe_allow_html=True)
        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), width="stretch")
        else:
            st.markdown('<div class="soft-note">confusion_matrix.png is not available yet.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-kicker">Discrimination</div><h3 class="card-title">ROC Curve</h3>', unsafe_allow_html=True)
        if ROC_CURVE_PATH.exists():
            st.image(str(ROC_CURVE_PATH), width="stretch")
        else:
            st.markdown('<div class="soft-note">roc_curve.png is not available yet.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_disclaimer():
    """渲染底部免责声明。"""
    st.markdown(
        """
        <div class="disclaimer">
            This tool is for educational purposes only and cannot replace professional medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Streamlit 应用入口。"""
    configure_page()
    render_hero()

    model, feature_columns = load_model_and_feature_columns()

    # 使用 session_state 保存最近一次预测结果，避免页面刷新后右侧结果立即丢失
    if "probability" not in st.session_state:
        st.session_state.probability = None
    if "risk_level" not in st.session_state:
        st.session_state.risk_level = None

    left_col, right_col = st.columns([1.08, 0.92], gap="large")
    with left_col:
        submitted, input_data = render_patient_card()

    if submitted:
        input_df = align_features(input_data, feature_columns)
        probability = predict_probability(model, input_df)
        st.session_state.probability = probability
        st.session_state.risk_level = classify_risk(probability)

    with right_col:
        render_result_card(st.session_state.probability, st.session_state.risk_level)

    render_model_explanation()
    render_model_performance()
    render_disclaimer()


if __name__ == "__main__":
    main()
