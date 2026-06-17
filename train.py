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

# 图片保存目录：用于保存最佳模型的混淆矩阵和 ROC 曲线
FIGURES_DIR = Path("figures")

# 模型保存目录：用于保存最佳模型和训练时的特征列名
MODELS_DIR = Path("models")

# 最佳模型保存路径
BEST_MODEL_PATH = MODELS_DIR / "best_diabetes_model.pkl"

# 训练时使用的原始特征列名保存路径
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"


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
    # Random Forest：基于多棵决策树的集成模型，常用于表格数据分类任务
    # Gradient Boosting：逐步提升的集成模型，通常能捕捉更复杂的非线性关系
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

        # 保存训练后的模型和预测结果，方便后面选择最佳模型和绘图
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

    return best_model_name, best_model, best_prediction, results_df


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
    best_model_name, best_model, best_prediction, _ = train_and_compare_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # 为最佳模型保存混淆矩阵和 ROC 曲线，方便后续查看模型表现
    save_confusion_matrix(y_test, best_prediction["y_pred"])
    save_roc_curve(y_test, best_prediction["y_proba"], best_model_name)

    # 保存最佳模型和训练时使用的特征列名
    save_artifacts(best_model, feature_columns)


if __name__ == "__main__":
    main()
