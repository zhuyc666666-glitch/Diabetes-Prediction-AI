from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# 最佳模型文件路径：该文件由 train.py 训练并保存得到
MODEL_PATH = Path("models") / "best_diabetes_model.pkl"

# 特征列名文件路径：用于保证网页输入数据和训练时的特征顺序一致
FEATURE_COLUMNS_PATH = Path("models") / "feature_columns.pkl"

# 页面中会采集的原始输入字段
RAW_INPUT_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]


def configure_page():
    """配置页面标题、布局和简洁医疗 AI 风格。"""
    # 设置 Streamlit 页面基础信息
    st.set_page_config(
        page_title="MedPredict AI",
        layout="centered",
    )

    # 使用 CSS 做轻量视觉优化：保持简洁、高级、偏医疗 AI 风格
    st.markdown(
        """
        <style>
        .stApp {
            background: #f5f9fb;
            color: #102a43;
        }
        .main .block-container {
            max-width: 880px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }
        h1 {
            color: #0b3d4a;
            font-weight: 760;
            letter-spacing: 0;
        }
        h2, h3 {
            color: #155e75;
            letter-spacing: 0;
        }
        .info-box {
            background: #ffffff;
            border: 1px solid #d8e8ed;
            border-left: 5px solid #0e7490;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            color: #334e68;
            box-shadow: 0 8px 24px rgba(15, 61, 74, 0.06);
        }
        .result-box {
            background: #ffffff;
            border: 1px solid #d8e8ed;
            border-radius: 8px;
            padding: 1.25rem;
            box-shadow: 0 8px 24px rgba(15, 61, 74, 0.07);
        }
        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            border: none;
            background: #0e7490;
            color: #ffffff;
            font-weight: 650;
            padding: 0.75rem 1rem;
        }
        div.stButton > button:hover {
            background: #155e75;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model_and_feature_columns():
    """加载训练好的最佳模型和训练时保存的特征列名。"""
    # 如果模型文件不存在，说明需要先运行 train.py 完成训练
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.info("Please run train.py first to generate the trained model.")
        st.stop()

    # 如果特征列名文件不存在，就无法可靠对齐预测输入
    if not FEATURE_COLUMNS_PATH.exists():
        st.error(f"Feature columns file not found: {FEATURE_COLUMNS_PATH}")
        st.info("Please run train.py first to generate feature_columns.pkl.")
        st.stop()

    # 加载模型；如果 train.py 保存的是 Pipeline，这里会同时加载预处理和分类器
    model = joblib.load(MODEL_PATH)

    # 加载训练时保存的特征列名
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    return model, feature_columns


def render_header():
    """渲染页面标题、副标题和使用说明。"""
    # 页面标题
    st.title("MedPredict AI")

    # 页面副标题
    st.subheader("Diabetes Risk Prediction System")

    # 页面说明：明确该工具仅用于教育用途
    st.markdown(
        """
        <div class="info-box">
        This tool is for educational purposes only and cannot replace professional medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_form():
    """渲染预测输入表单，并返回用户输入。"""
    st.markdown("### Patient Information")

    # 使用 Streamlit 表单，让用户填写完所有字段后再统一提交预测
    with st.form("diabetes_prediction_form"):
        left_col, right_col = st.columns(2)

        with left_col:
            # gender 是类别变量，训练时会进行 One-Hot Encoding
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])

            # age 是年龄，使用滑块让输入更直观
            age = st.slider("Age", min_value=0.0, max_value=100.0, value=45.0, step=1.0)

            # hypertension 是二分类变量，0 表示无高血压，1 表示有高血压
            hypertension = st.selectbox(
                "Hypertension",
                [0, 1],
                format_func=lambda value: "Yes" if value == 1 else "No",
            )

            # heart_disease 是二分类变量，0 表示无心脏病，1 表示有心脏病
            heart_disease = st.selectbox(
                "Heart Disease",
                [0, 1],
                format_func=lambda value: "Yes" if value == 1 else "No",
            )

        with right_col:
            # smoking_history 是类别变量，训练时会进行 One-Hot Encoding
            smoking_history = st.selectbox(
                "Smoking History",
                ["never", "No Info", "current", "former", "ever", "not current"],
            )

            # bmi 是身体质量指数
            bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=27.5, step=0.1)

            # HbA1c_level 是糖化血红蛋白水平
            hba1c_level = st.number_input(
                "HbA1c Level",
                min_value=3.0,
                max_value=15.0,
                value=5.8,
                step=0.1,
            )

            # blood_glucose_level 是血糖水平
            blood_glucose_level = st.number_input(
                "Blood Glucose Level",
                min_value=50,
                max_value=400,
                value=140,
                step=1,
            )

        # 预测按钮，只有点击后才执行模型预测
        submitted = st.form_submit_button("Predict Diabetes Risk")

    # 将用户输入整理成一条样本，字段名与原始训练数据保持一致
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
    """将用户输入和训练时保存的 feature_columns 对齐。"""
    # 先构造原始输入 DataFrame，适用于保存完整 Pipeline 的模型
    raw_df = pd.DataFrame([input_data])

    # 情况一：feature_columns 保存的是原始训练特征列名
    # 当前 train.py 保存的就是这种格式，模型 Pipeline 会在内部完成 One-Hot Encoding
    if set(feature_columns).issubset(set(raw_df.columns)):
        return raw_df.reindex(columns=feature_columns)

    # 情况二：feature_columns 保存的是 One-Hot Encoding 后的特征列名
    # 这里手动进行 get_dummies，再按训练时的列名补齐和排序
    encoded_df = pd.get_dummies(
        raw_df,
        columns=["gender", "smoking_history"],
        dtype=int,
    )

    # reindex 会按训练时列名排序；缺失列用 0 补齐，表示该 One-Hot 类别未被选中
    aligned_df = encoded_df.reindex(columns=feature_columns, fill_value=0)

    return aligned_df


def classify_risk(probability):
    """根据糖尿病预测概率输出风险等级。"""
    # probability < 0.3: Low Risk
    if probability < 0.3:
        return "Low Risk"

    # 0.3 <= probability < 0.7: Medium Risk
    if probability < 0.7:
        return "Medium Risk"

    # probability >= 0.7: High Risk
    return "High Risk"


def render_result(probability, risk_level):
    """展示 Diabetes probability 和 Prediction result。"""
    st.markdown("### Prediction Result")

    # 使用简洁结果卡片突出概率和风险等级
    st.markdown(
        f"""
        <div class="result-box">
            <h3>Diabetes probability</h3>
            <p style="font-size: 2rem; font-weight: 760; margin: 0 0 1rem 0;">
                {probability:.2%}
            </p>
            <h3>Prediction result</h3>
            <p style="font-size: 1.5rem; font-weight: 720; margin: 0;">
                {risk_level}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def predict_probability(model, input_df):
    """使用模型预测 diabetes=1 的概率。"""
    # 优先使用 predict_proba 获取阳性类别概率
    if hasattr(model, "predict_proba"):
        return model.predict_proba(input_df)[0, 1]

    # 如果模型不支持 predict_proba，则给出清晰错误提示
    st.error("The loaded model does not support probability prediction.")
    st.stop()


def main():
    """Streamlit 应用入口函数。"""
    # 配置页面和视觉样式
    configure_page()

    # 渲染标题、副标题和说明
    render_header()

    # 加载模型和训练时使用的特征列
    model, feature_columns = load_model_and_feature_columns()

    # 渲染输入表单
    submitted, input_data = render_input_form()

    # 点击 Predict Diabetes Risk 后执行预测
    if submitted:
        # 将输入数据与训练时特征列对齐
        input_df = align_features(input_data, feature_columns)

        # 预测糖尿病概率
        probability = predict_probability(model, input_df)

        # 根据概率进行风险分层
        risk_level = classify_risk(probability)

        # 输出预测概率和风险结果
        render_result(probability, risk_level)


if __name__ == "__main__":
    main()
