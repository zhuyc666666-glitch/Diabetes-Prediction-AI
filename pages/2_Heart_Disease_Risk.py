import streamlit as st


def configure_page():
    """配置心脏病风险页面和统一视觉样式。"""
    st.set_page_config(
        page_title="Heart Disease Risk | MedPredict AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Coming Soon 页面也保持和平台首页一致的 Apple 医疗 AI 风格。
    st.markdown(
        """
        <style>
        :root {
            --bg: #f7fbfc;
            --ink: #102a43;
            --muted: #5f7384;
            --line: rgba(14, 116, 144, 0.14);
            --teal: #0e7490;
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
            max-width: 1180px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0);
        }

        .hero {
            padding: 3.6rem 1rem 2.2rem;
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
            margin: 1rem 0 0.5rem;
            font-size: clamp(2.8rem, 6vw, 5.2rem);
            line-height: 0.98;
            font-weight: 790;
            color: #082f3b;
        }

        .hero p {
            max-width: 720px;
            margin: 0 auto;
            color: var(--muted);
            font-size: 1.08rem;
            line-height: 1.7;
        }

        .glass-card {
            padding: 1.6rem;
            border-radius: 28px;
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid rgba(14, 116, 144, 0.13);
            box-shadow: var(--shadow);
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

        .card-copy {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.65;
        }

        .roadmap-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1.2rem;
        }

        @media (max-width: 900px) {
            .roadmap-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page():
    """渲染心脏病风险 Coming Soon 页面。"""
    st.markdown(
        """
        <section class="hero">
            <div class="hero-pill">Coming Soon</div>
            <h1>Heart Disease Risk Prediction</h1>
            <p>
                A planned cardiovascular risk assessment module for future model training,
                evaluation, and explainable clinical decision support.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # 这里先展示未来功能规划，不训练模型，也不读取新的数据文件。
    st.markdown(
        """
        <div class="roadmap-grid">
            <div class="glass-card">
                <div class="card-kicker">Input Design</div>
                <h3 class="card-title">Clinical Profile</h3>
                <p class="card-copy">Future inputs may include age, blood pressure, cholesterol, ECG results, exercise response, and chest pain patterns.</p>
            </div>
            <div class="glass-card">
                <div class="card-kicker">Modeling</div>
                <h3 class="card-title">Risk Classifier</h3>
                <p class="card-copy">The module can later compare logistic regression, tree-based models, and gradient boosting classifiers.</p>
            </div>
            <div class="glass-card">
                <div class="card-kicker">Interpretability</div>
                <h3 class="card-title">Explainable Output</h3>
                <p class="card-copy">Planned explanations include feature importance, probability bands, and visual model performance summaries.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """页面入口函数。"""
    configure_page()
    render_page()


if __name__ == "__main__":
    main()
