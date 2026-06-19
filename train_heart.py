from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
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


# 数据文件路径：严格按照要求，只使用 combined 版本的心脏病数据集。
DATA_PATH = Path("data") / "heart_disease_combined.csv"

# 图片保存目录：用于保存心脏病模型的评估图和特征重要性图。
FIGURES_DIR = Path("figures")

# 模型保存目录：用于保存最佳模型、原始特征列名和特征重要性 CSV。
MODELS_DIR = Path("models")

# 心脏病最佳模型保存路径。
BEST_MODEL_PATH = MODELS_DIR / "best_heart_model.pkl"

# 心脏病模型训练时使用的原始特征列名保存路径。
FEATURE_COLUMNS_PATH = MODELS_DIR / "heart_feature_columns.pkl"

# 心脏病模型特征重要性 CSV 保存路径。
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "heart_feature_importance.csv"

# 心脏病模型评估图保存路径。
CONFUSION_MATRIX_PATH = FIGURES_DIR / "heart_confusion_matrix.png"
ROC_CURVE_PATH = FIGURES_DIR / "heart_roc_curve.png"
FEATURE_IMPORTANCE_FIGURE_PATH = FIGURES_DIR / "heart_feature_importance.png"

# 目标列候选名称，顺序就是自动识别时的优先级。
TARGET_CANDIDATES = [
    "heart_disease",
    "target",
    "HeartDisease",
    "num",
    "diagnosis",
    "condition",
]

# 明显无用的身份信息列，训练前会自动删除。
USELESS_COLUMNS = ["id", "patient_id", "name"]


def load_data():
    """读取心脏病 combined 数据集。"""
    # 如果文件不存在，直接抛出清晰错误，避免误读 heart.csv 或 cleveland 文件。
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    # 只读取 data/heart_disease_combined.csv，不读取其他心脏病数据文件。
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data from: {DATA_PATH}")
    print(f"Original data shape: {df.shape}")

    return df


def find_target_column(df):
    """按照指定优先级自动识别目标列。"""
    # 先做精确匹配，保证 heart_disease 优先于 target 等其他候选列。
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate

    # 如果列名大小写不完全一致，再做忽略大小写的匹配。
    lower_to_original = {column.lower(): column for column in df.columns}
    for candidate in TARGET_CANDIDATES:
        matched_column = lower_to_original.get(candidate.lower())
        if matched_column is not None:
            return matched_column

    raise ValueError(f"No target column found. Candidates: {TARGET_CANDIDATES}")


def drop_useless_columns(df, target_column):
    """删除 id、patient_id、name 等明显无用列。"""
    # 用小写映射做匹配，这样 ID、Patient_ID、Name 等大小写变体也能识别。
    lower_to_original = {column.lower(): column for column in df.columns}
    columns_to_drop = []

    for useless_column in USELESS_COLUMNS:
        original_column = lower_to_original.get(useless_column.lower())
        if original_column is not None and original_column != target_column:
            columns_to_drop.append(original_column)

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"Dropped useless columns: {columns_to_drop}")
    else:
        print("No useless columns found.")

    return df


def split_features_and_label(df):
    """自动识别目标列，并拆分特征 X 和标签 y。"""
    # 自动找到目标列。
    target_column = find_target_column(df)
    print(f"Detected target column: {target_column}")

    # 删除明显无用列，避免身份信息影响模型训练。
    df = drop_useless_columns(df, target_column)

    # 如果目标列是 num，则把多分类严重程度转换为二分类标签。
    if target_column.lower() == "num":
        y = (df[target_column] > 0).astype(int)
        print("Converted num to binary label: num=0 -> 0, num>0 -> 1")
    else:
        y = df[target_column]

    # 确保标签是整数二分类形式，便于计算分类指标和 ROC-AUC。
    y = y.astype(int)

    # 目标列不能作为模型输入特征。
    X = df.drop(columns=[target_column])

    print("Target value counts:")
    print(y.value_counts().sort_index())
    print(f"Feature shape: {X.shape}")

    return X, y, target_column


def get_feature_types(X):
    """自动区分类别变量和数值变量。"""
    # object 类型作为类别变量；同时兼容 pandas 新版本的 string 字符串类型。
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

    # 数值型变量作为数值变量。
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    return numeric_features, categorical_features


def create_one_hot_encoder():
    """创建 One-Hot Encoder，并兼容不同版本的 scikit-learn。"""
    # 新版本 scikit-learn 使用 sparse_output，旧版本使用 sparse。
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X):
    """构建数值变量和类别变量的预处理流程。"""
    numeric_features, categorical_features = get_feature_types(X)

    # 数值变量缺失值用中位数填补。
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # 类别变量缺失值用众数填补，再做 One-Hot Encoding。
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", create_one_hot_encoder()),
        ]
    )

    # ColumnTransformer 将不同类型的列送入不同预处理流程。
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def get_candidate_models():
    """定义需要训练和比较的三个候选模型。"""
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def build_pipeline(preprocessor, classifier):
    """将预处理器和分类器封装成完整 Pipeline。"""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_predictions(y_test, y_pred, y_proba):
    """计算 Accuracy、Precision、Recall、F1-score 和 ROC-AUC。"""
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """训练三个模型，输出指标和 Classification Report，并选择最佳模型。"""
    models = get_candidate_models()
    results = []
    trained_models = {}
    predictions = {}

    for model_name, classifier in models.items():
        print("\n" + "=" * 80)
        print(f"Training model: {model_name}")
        print("=" * 80)

        # 每个模型使用相同的预处理逻辑，保证比较公平。
        preprocessor = build_preprocessor(X_train)
        model = build_pipeline(preprocessor, classifier)

        # 训练当前模型。
        model.fit(X_train, y_train)

        # 在测试集上预测类别和阳性类别概率。
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # 计算当前模型的评估指标。
        metrics = evaluate_predictions(y_test, y_pred, y_proba)
        metrics["Model"] = model_name
        results.append(metrics)

        # 保存模型和预测结果，后续用于选择最佳模型和绘图。
        trained_models[model_name] = model
        predictions[model_name] = {
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    # 用表格形式打印三个模型结果。
    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    ]

    print("\n" + "=" * 80)
    print("Model Comparison Results")
    print("=" * 80)
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    # 自动选择 ROC-AUC 最高的模型作为最佳模型。
    best_row = results_df.loc[results_df["ROC-AUC"].idxmax()]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]
    best_prediction = predictions[best_model_name]

    print(f"\nBest model selected by ROC-AUC: {best_model_name}")
    print(f"Best ROC-AUC: {best_row['ROC-AUC']:.4f}")

    return best_model_name, best_model, best_prediction, results_df, trained_models


def clean_feature_name(feature_name):
    """清理 Pipeline 输出的特征名，方便图表和 CSV 阅读。"""
    cleaned_name = feature_name.replace("numeric__", "").replace("categorical__", "")
    cleaned_name = cleaned_name.replace("onehot__", "")
    return cleaned_name


def get_model_for_feature_importance(best_model_name, best_model, trained_models):
    """选择可用于 feature_importances_ 分析的树模型。"""
    best_classifier = best_model.named_steps["classifier"]
    if hasattr(best_classifier, "feature_importances_"):
        return best_model_name, best_model

    # 如果最佳模型是 Logistic Regression，则回退到 Random Forest 做特征重要性分析。
    random_forest_model = trained_models.get("Random Forest")
    if random_forest_model is not None:
        random_forest_classifier = random_forest_model.named_steps["classifier"]
        if hasattr(random_forest_classifier, "feature_importances_"):
            print(
                "\nBest model has no feature_importances_. "
                "Use Random Forest for feature importance analysis."
            )
            return "Random Forest", random_forest_model

    return None, None


def save_feature_importance(best_model_name, best_model, trained_models):
    """保存特征重要性 CSV 和图片。"""
    importance_model_name, importance_model = get_model_for_feature_importance(
        best_model_name,
        best_model,
        trained_models,
    )

    if importance_model is None:
        print("\nNo model with feature_importances_ found. Skip feature importance.")
        return

    preprocessor = importance_model.named_steps["preprocessor"]
    classifier = importance_model.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        print("\nSelected model has no feature_importances_. Skip feature importance.")
        return

    # 获取预处理后的特征名，确保和 feature_importances_ 一一对应。
    transformed_feature_names = preprocessor.get_feature_names_out()
    cleaned_feature_names = [
        clean_feature_name(feature_name) for feature_name in transformed_feature_names
    ]

    importance_df = pd.DataFrame(
        {
            "feature": cleaned_feature_names,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    print(f"Feature importance CSV saved to: {FEATURE_IMPORTANCE_PATH}")

    # 保存 Top 15 特征重要性图。
    top_features = importance_df.head(15).sort_values("importance", ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(top_features["feature"], top_features["importance"], color="#0e7490")
    plt.title(f"Heart Disease Feature Importance ({importance_model_name})")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_FIGURE_PATH, dpi=300)
    plt.close()
    print(f"Feature importance figure saved to: {FEATURE_IMPORTANCE_FIGURE_PATH}")


def save_confusion_matrix(y_test, y_pred):
    """绘制并保存最佳心脏病模型的混淆矩阵。"""
    matrix = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1])
    display.plot(cmap="Blues", values_format="d")
    plt.title("Heart Disease Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


def save_roc_curve(y_test, y_proba, model_name):
    """绘制并保存最佳心脏病模型的 ROC 曲线。"""
    RocCurveDisplay.from_predictions(y_test, y_proba, name=model_name)
    plt.title("Heart Disease ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=300)
    plt.close()
    print(f"ROC curve saved to: {ROC_CURVE_PATH}")


def save_artifacts(best_model, feature_columns):
    """保存最佳模型和训练时使用的原始特征列。"""
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"Best model saved to: {BEST_MODEL_PATH}")

    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
    print(f"Feature columns saved to: {FEATURE_COLUMNS_PATH}")


def main():
    """主函数：完成心脏病模型训练、比较、绘图和保存。"""
    # 自动创建输出目录。
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 读取固定数据文件。
    df = load_data()

    # 拆分 X 和 y，并按要求处理 num 目标列。
    X, y, _ = split_features_and_label(df)

    # 保存训练时使用的原始特征列名。
    feature_columns = X.columns.tolist()

    # 按要求划分训练集和测试集，stratify=y 保持类别比例一致。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 训练并比较三个模型。
    best_model_name, best_model, best_prediction, _, trained_models = (
        train_and_compare_models(
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    # 保存最佳模型的评估图。
    save_confusion_matrix(y_test, best_prediction["y_pred"])
    save_roc_curve(y_test, best_prediction["y_proba"], best_model_name)

    # 保存最佳模型和原始特征列名。
    save_artifacts(best_model, feature_columns)

    # 保存特征重要性 CSV 和图片。
    save_feature_importance(best_model_name, best_model, trained_models)


if __name__ == "__main__":
    main()
