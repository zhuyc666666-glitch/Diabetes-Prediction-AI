from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# 数据文件路径：按照要求读取 data 文件夹下的 diabetes_prediction_dataset.csv
DATA_PATH = Path("data") / "diabetes_prediction_dataset.csv"

# 图片保存目录：用于保存模型评估图和特征重要性图
FIGURES_DIR = Path("figures")

# 模型保存目录：用于保存最佳模型、特征列名和特征重要性 CSV
MODELS_DIR = Path("models")

# 最佳模型保存路径
BEST_MODEL_PATH = MODELS_DIR / "best_diabetes_model.pkl"

# 训练时使用的原始特征列名保存路径
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"

# 特征重要性 CSV 保存路径
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"

# 特征重要性图片保存路径
FEATURE_IMPORTANCE_FIGURE_PATH = FIGURES_DIR / "feature_importance.png"


def load_data():
    """读取糖尿病数据集，并删除重复值。"""
    # 检查数据文件是否存在，避免路径错误时出现不清晰的报错
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    # 使用 pandas 读取 CSV 文件
    df = pd.read_csv(DATA_PATH)

    # 删除重复行，减少重复样本对模型训练和评估结果的影响
    before_drop = df.shape[0]
    df = df.drop_duplicates()
    after_drop = df.shape[0]
    print(f"Removed duplicate rows: {before_drop - after_drop}")

    return df


def split_features_and_label(df):
    """将 diabetes 作为标签 y，其余列作为特征 X。"""
    # diabetes 是本项目要预测的目标列，必须存在
    if "diabetes" not in df.columns:
        raise ValueError("Column not found: diabetes")

    # y 表示标签，也就是模型要学习预测的糖尿病结果
    y = df["diabetes"]

    # X 表示特征，删除 diabetes 后剩余的列全部作为输入特征
    X = df.drop(columns=["diabetes"])

    return X, y


def build_preprocessor():
    """构建特征预处理器，对类别变量做 One-Hot Encoding。"""
    # 按要求对 gender 和 smoking_history 两个类别特征进行 One-Hot Encoding
    categorical_features = ["gender", "smoking_history"]

    # 不同 scikit-learn 版本的 OneHotEncoder 参数名略有差异：
    # 新版本使用 sparse_output，旧版本使用 sparse。
    # 这里通过 try/except 做兼容，同时强制输出密集矩阵，方便三个模型统一使用。
    try:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    # ColumnTransformer 可以只处理指定列，并保留其他数值列
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                one_hot_encoder,
                categorical_features,
            )
        ],
        # 其他数值型特征不做额外处理，直接传入模型
        remainder="passthrough",
    )

    return preprocessor


def get_candidate_models():
    """定义需要训练和比较的三个候选模型。"""
    # Logistic Regression：线性分类模型，作为简单且可解释的基线模型
    # Random Forest：基于多棵决策树的集成模型，适合做特征重要性分析
    # Gradient Boosting：逐步提升的集成模型，也支持 feature_importances_
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def build_pipeline(classifier):
    """将预处理步骤和分类器封装成一个完整 Pipeline。"""
    # 每个模型都使用相同的预处理流程，保证比较公平
    preprocessor = build_preprocessor()

    # Pipeline 可以把 One-Hot Encoding 和模型训练串起来，保存模型时也会一起保存预处理逻辑
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_predictions(y_test, y_pred, y_proba):
    """根据预测结果计算常用分类指标。"""
    # zero_division=0 可以避免模型没有预测出某一类时 precision/recall 报警
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """训练三个模型，输出各自指标，并返回 ROC-AUC 最高的最佳模型。"""
    models = get_candidate_models()
    results = []
    trained_models = {}
    predictions = {}

    # 依次训练每个候选模型，并在同一个测试集上评估
    for model_name, classifier in models.items():
        print(f"\nTraining model: {model_name}")

        # 为当前分类器构建独立 Pipeline
        model = build_pipeline(classifier)

        # 使用训练集训练模型
        model.fit(X_train, y_train)

        # 在测试集上预测类别
        y_pred = model.predict(X_test)

        # 获取 diabetes=1 的预测概率，用于计算 ROC-AUC 和绘制 ROC 曲线
        y_proba = model.predict_proba(X_test)[:, 1]

        # 计算当前模型的评价指标
        metrics = evaluate_predictions(y_test, y_pred, y_proba)
        metrics["Model"] = model_name
        results.append(metrics)

        # 保存训练后的模型和预测结果，方便后面选择最佳模型、绘图和分析特征重要性
        trained_models[model_name] = model
        predictions[model_name] = {
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

        # 输出当前模型的 Classification Report，便于观察每个类别的表现
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    # 用 DataFrame 以表格形式展示三个模型的指标
    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    ]

    print("\nModel Comparison Results:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    # 按 ROC-AUC 自动选择最佳模型
    best_row = results_df.loc[results_df["ROC-AUC"].idxmax()]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]
    best_prediction = predictions[best_model_name]

    print(f"\nBest model selected by ROC-AUC: {best_model_name}")
    print(f"Best ROC-AUC: {best_row['ROC-AUC']:.4f}")

    return best_model_name, best_model, best_prediction, results_df, trained_models


def clean_feature_name(feature_name):
    """清理 Pipeline 输出的特征名，让图表和网页展示更易读。"""
    # ColumnTransformer 会给特征名加上 categorical__ 或 remainder__ 前缀，这里去掉前缀
    cleaned_name = feature_name.replace("categorical__", "").replace("remainder__", "")

    # One-Hot Encoding 后的特征名一般形如 gender_Male 或 smoking_history_never
    return cleaned_name


def get_model_for_feature_importance(best_model_name, best_model, trained_models):
    """选择用于特征重要性分析的树模型。"""
    # 优先使用最佳模型；如果最佳模型是 Random Forest 或 Gradient Boosting，通常会有 feature_importances_
    best_classifier = best_model.named_steps["classifier"]
    if hasattr(best_classifier, "feature_importances_"):
        return best_model_name, best_model

    # 如果最佳模型是 Logistic Regression，没有 feature_importances_，则退回使用 Random Forest
    random_forest_model = trained_models.get("Random Forest")
    if random_forest_model is not None:
        random_forest_classifier = random_forest_model.named_steps["classifier"]
        if hasattr(random_forest_classifier, "feature_importances_"):
            print(
                "\nBest model has no feature_importances_. "
                "Use Random Forest for feature importance analysis."
            )
            return "Random Forest", random_forest_model

    # 如果没有任何可用树模型，则跳过特征重要性分析
    return None, None


def save_feature_importance(best_model_name, best_model, trained_models):
    """提取、排序并保存树模型的 Feature Importance。"""
    # 选择最佳树模型；如果最佳模型不是树模型，则使用 Random Forest 作为解释模型
    importance_model_name, importance_model = get_model_for_feature_importance(
        best_model_name,
        best_model,
        trained_models,
    )

    if importance_model is None:
        print("\nNo model with feature_importances_ found. Skip feature importance.")
        return

    # 从 Pipeline 中取出预处理器和分类器
    preprocessor = importance_model.named_steps["preprocessor"]
    classifier = importance_model.named_steps["classifier"]

    # 如果分类器没有 feature_importances_ 属性，则跳过
    if not hasattr(classifier, "feature_importances_"):
        print("\nSelected model has no feature_importances_. Skip feature importance.")
        return

    # 获取 One-Hot Encoding 后的特征名，确保和 feature_importances_ 一一对应
    transformed_feature_names = preprocessor.get_feature_names_out()
    cleaned_feature_names = [
        clean_feature_name(feature_name) for feature_name in transformed_feature_names
    ]

    # 整理成 DataFrame，并按重要性从高到低排序
    importance_df = pd.DataFrame(
        {
            "feature": cleaned_feature_names,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    # 保存完整特征重要性表格，供 Streamlit 页面读取 Top 5 Risk Factors
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    print(f"Feature importance CSV saved to: {FEATURE_IMPORTANCE_PATH}")

    # 绘制 Top 15 特征重要性水平柱状图，让图表保持清晰
    top_features = importance_df.head(15).sort_values("importance", ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(top_features["feature"], top_features["importance"], color="#0e7490")
    plt.title(f"Feature Importance ({importance_model_name})")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_FIGURE_PATH, dpi=300)
    plt.close()
    print(f"Feature importance figure saved to: {FEATURE_IMPORTANCE_FIGURE_PATH}")


def save_confusion_matrix(y_test, y_pred):
    """绘制并保存最佳模型的混淆矩阵。"""
    # 混淆矩阵用于观察真实标签和预测标签之间的对应关系
    matrix = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1])
    display.plot(cmap="Blues", values_format="d")

    # 图表标题和坐标轴使用英文
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300)
    plt.close()


def save_roc_curve(y_test, y_proba, model_name):
    """绘制并保存最佳模型的 ROC 曲线。"""
    # ROC 曲线展示模型在不同分类阈值下的表现
    RocCurveDisplay.from_predictions(y_test, y_proba, name=model_name)

    # 图表标题和坐标轴使用英文
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=300)
    plt.close()


def save_artifacts(best_model, feature_columns):
    """保存最佳模型和训练时使用的原始特征列名。"""
    # 保存完整 Pipeline，后续预测时可以直接输入原始特征列
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"\nBest model saved to: {BEST_MODEL_PATH}")

    # 保存训练时使用的特征列名，部署或预测时可用于校验输入字段是否一致
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
    print(f"Feature columns saved to: {FEATURE_COLUMNS_PATH}")


def main():
    """主函数：完成数据读取、训练比较、最佳模型选择、绘图和保存。"""
    # 自动创建 figures 和 models 文件夹，如果已经存在则不会报错
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 读取数据并删除重复值
    df = load_data()

    # 将 diabetes 作为标签 y，其余列作为特征 X
    X, y = split_features_and_label(df)

    # 保存训练时使用的原始特征列名
    feature_columns = X.columns.tolist()

    # 使用 train_test_split 划分训练集和测试集
    # stratify=y 可以保持训练集和测试集中 diabetes=0/1 的比例更接近原始数据
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 训练并比较 Logistic Regression、Random Forest、Gradient Boosting 三个模型
    best_model_name, best_model, best_prediction, _, trained_models = (
        train_and_compare_models(
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    # 为最佳模型保存混淆矩阵和 ROC 曲线，方便后续查看模型表现
    save_confusion_matrix(y_test, best_prediction["y_pred"])
    save_roc_curve(y_test, best_prediction["y_proba"], best_model_name)

    # 保存最佳模型和训练时使用的特征列名
    save_artifacts(best_model, feature_columns)

    # 保存树模型的 Feature Importance 图和 CSV
    save_feature_importance(best_model_name, best_model, trained_models)


if __name__ == "__main__":
    main()
