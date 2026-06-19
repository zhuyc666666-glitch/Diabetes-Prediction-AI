from pathlib import Path

import streamlit as st


# 图表路径保持不变，均由 train.py 生成。
CONFUSION_MATRIX_PATH = Path("figures") / "confusion_matrix.png"
ROC_CURVE_PATH = Path("figures") / "roc_curve.png"
FEATURE_IMPORTANCE_PATH = Path("figures") / "feature_importance.png"


def configure_page():
    """配置模型解释页面和统一视觉样式。"""
    st.set_page_config(
        page_title="Model Explanation | MedPredict AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 使用 CSS 保持多页面平台视觉一致。
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
            max-width: 1280px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0);
        }

        .hero {
            padding: 2.8rem 1rem 1.7rem;
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
            max-width: 760px;
            margin: 0 auto;
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.7;
        }

        .glass-card {
            padding: 1.35rem;
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
            margin: 0 0 0.9rem;
            color: #0f3d4c;
            font-size: 1.45rem;
            line-height: 1.2;
            font-weight: 760;
        }

        .soft-note {
            color: var(--muted);
            background: rgba(255, 255, 255, 0.74);
            border: 1px dashed rgba(14, 116, 144, 0.25);
            border-radius: 18px;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_figure_card(title, kicker, image_path):
    """展示单张模型解释图片；如果图片不存在则显示提示。"""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="card-kicker">{kicker}</div><h3 class="card-title">{title}</h3>',
        unsafe_allow_html=True,
    )

    # 图片不存在时不抛出异常，保持部署页面稳定。
    if image_path.exists():
        st.image(str(image_path), width="stretch")
    else:
        st.markdown(
            f'<div class="soft-note">{image_path.name} is not available yet. Please run train.py first.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_page():
    """渲染模型解释页面。"""
    st.markdown(
        """
        <section class="hero">
            <div class="hero-pill">Explainable AI</div>
            <h1>Model Explanation</h1>
            <p>
                Review the trained diabetes model through evaluation charts and
                interpretable feature importance results.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # 使用宽屏两列布局展示评估图，下面单独展示特征重要性图。
    left, right = st.columns(2, gap="large")
    with left:
        render_figure_card("Confusion Matrix", "Evaluation", CONFUSION_MATRIX_PATH)
    with right:
        render_figure_card("ROC Curve", "Discrimination", ROC_CURVE_PATH)

    st.write("")
    render_figure_card("Feature Importance", "Interpretability", FEATURE_IMPORTANCE_PATH)


def main():
    """页面入口函数。"""
    configure_page()
    render_page()


if __name__ == "__main__":
    main()
