import streamlit as st


def configure_page():
    """配置 About 页面和统一视觉样式。"""
    st.set_page_config(
        page_title="About | MedPredict AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # About 页面继续使用平台统一的白色、浅蓝绿色和圆角卡片风格。
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
            max-width: 1120px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0);
        }

        .hero {
            padding: 3.2rem 1rem 2rem;
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

        .glass-card {
            padding: 1.65rem;
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
            font-size: 1.5rem;
            line-height: 1.2;
            font-weight: 760;
        }

        .card-copy {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.75;
        }

        .disclaimer {
            margin-top: 1.4rem;
            padding: 1.1rem 1.25rem;
            border-radius: 22px;
            text-align: center;
            color: #516778;
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid rgba(14, 116, 144, 0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page():
    """渲染项目介绍页面。"""
    st.markdown(
        """
        <section class="hero">
            <div class="hero-pill">About the Project</div>
            <h1>MedPredict AI</h1>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # 项目背景说明，定位为医学 AI 本科生作品集项目。
    st.markdown(
        """
        <div class="glass-card">
            <div class="card-kicker">Project Background</div>
            <h3 class="card-title">A medical AI undergraduate portfolio project</h3>
            <p class="card-copy">
                MedPredict AI is a multi-page medical risk assessment platform built with Streamlit.
                The current working module focuses on diabetes risk prediction using structured patient
                information, supervised machine learning, model evaluation metrics, and interpretable
                feature importance analysis.
            </p>
            <p class="card-copy">
                The project demonstrates an end-to-end undergraduate medical AI workflow:
                exploratory data analysis, model training, evaluation visualization, deployment-ready
                interface design, and responsible communication of educational limitations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="disclaimer">
            This platform is for educational purposes only and cannot replace professional medical diagnosis.
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
