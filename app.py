import streamlit as st


def configure_page():
    """配置首页页面信息和全局视觉样式。"""
    st.set_page_config(
        page_title="MedPredict AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 使用自定义 CSS 统一平台首页视觉：
    # Apple 风、白色背景、浅蓝绿色渐变、圆角卡片和轻微阴影。
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
                radial-gradient(circle at top left, rgba(125, 211, 252, 0.28), transparent 34rem),
                radial-gradient(circle at top right, rgba(94, 234, 212, 0.22), transparent 32rem),
                linear-gradient(180deg, #ffffff 0%, var(--bg) 100%);
            color: var(--ink);
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
        }

        .main .block-container {
            max-width: 1280px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0);
        }

        h1, h2, h3, p {
            letter-spacing: 0;
        }

        .hero {
            padding: 4.2rem 1rem 3.2rem;
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
            font-size: clamp(3.2rem, 7vw, 6.3rem);
            line-height: 0.95;
            font-weight: 790;
            color: #082f3b;
        }

        .hero h2 {
            margin: 0.5rem auto 0;
            max-width: 760px;
            color: #0e7490;
            font-size: clamp(1.3rem, 2.5vw, 2.05rem);
            line-height: 1.25;
            font-weight: 620;
        }

        .platform-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1.35rem;
            margin-top: 1.3rem;
        }

        .feature-card {
            min-height: 245px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 1.55rem;
            border-radius: 28px;
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid rgba(14, 116, 144, 0.13);
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }

        .feature-kicker {
            color: var(--teal);
            font-size: 0.8rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
        }

        .feature-card h3 {
            margin: 0;
            color: #0f3d4c;
            font-size: 1.45rem;
            line-height: 1.2;
            font-weight: 760;
        }

        .feature-card p {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.6;
        }

        .feature-link {
            margin-top: 1rem;
            color: #0e7490;
            font-weight: 760;
        }

        .footer-note {
            margin-top: 2rem;
            padding: 1rem 1.2rem;
            border-radius: 22px;
            text-align: center;
            color: #516778;
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid rgba(14, 116, 144, 0.12);
        }

        @media (max-width: 900px) {
            .platform-grid {
                grid-template-columns: 1fr;
            }
            .hero {
                padding-top: 2.5rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_home():
    """渲染 Medical AI Platform 首页。"""
    st.markdown(
        """
        <section class="hero">
            <div class="hero-pill">Medical AI Platform</div>
            <h1>MedPredict AI</h1>
            <h2>An AI-powered medical risk assessment platform</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # 首页用三张功能卡片承载平台入口，避免普通表单式页面观感。
    st.markdown(
        """
        <div class="platform-grid">
            <div class="feature-card">
                <div>
                    <div class="feature-kicker">Risk Prediction</div>
                    <h3>Diabetes Risk Prediction</h3>
                    <p>Estimate diabetes risk from clinical indicators and lifestyle information using a trained machine learning model.</p>
                </div>
                <div class="feature-link">Open from sidebar: 1 Diabetes Risk</div>
            </div>
            <div class="feature-card">
                <div>
                    <div class="feature-kicker">Coming Soon</div>
                    <h3>Heart Disease Risk Prediction</h3>
                    <p>A future module for cardiovascular risk screening with structured patient inputs and interpretable model output.</p>
                </div>
                <div class="feature-link">Open from sidebar: 2 Heart Disease Risk</div>
            </div>
            <div class="feature-card">
                <div>
                    <div class="feature-kicker">Explainable AI</div>
                    <h3>Explainable AI</h3>
                    <p>Review model evaluation figures, ROC performance, confusion matrix results, and feature importance analysis.</p>
                </div>
                <div class="feature-link">Open from sidebar: 3 Model Explanation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="footer-note">
            This platform is designed as a clean medical AI portfolio project for education and demonstration.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """首页入口函数。"""
    configure_page()
    render_home()


if __name__ == "__main__":
    main()
