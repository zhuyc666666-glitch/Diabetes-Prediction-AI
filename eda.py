from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# 数据文件路径：统一读取 data 文件夹下的 diabetes_prediction_dataset.csv
CSV_PATH = Path("data") / "diabetes_prediction_dataset.csv"

# 图片输出文件夹路径：所有探索性分析图表都会保存到 figures 文件夹
FIGURES_DIR = Path("figures")


def print_basic_info(df):
    """打印数据集的基础信息。"""
    # 1. 查看数据前 5 行，快速了解字段和样本内容
    print("1. First 5 rows:")
    print(df.head())

    # 2. 查看数据维度，格式为：(行数, 列数)
    print("\n2. Dataset shape:")
    print(df.shape)

    # 3. 查看每一列的数据类型，判断哪些是数值变量、哪些是类别变量
    print("\n3. Column data types:")
    print(df.dtypes)

    # 4. 统计每一列缺失值数量，用于判断是否需要清洗数据
    print("\n4. Missing values:")
    print(df.isnull().sum())

    # 5. 统计 diabetes 标签中 0 和 1 的数量，观察类别是否平衡
    print("\n5. Counts of diabetes values:")
    if "diabetes" in df.columns:
        print(df["diabetes"].value_counts().sort_index())
    else:
        print("Column not found: diabetes")


def save_histogram(df, column, title, xlabel, output_file):
    """绘制并保存单个数值列的分布直方图。"""
    # 如果目标列不存在，就跳过绘图，避免程序直接报错中断
    if column not in df.columns:
        print(f"Column not found, skip plot: {column}")
        return

    # 创建画布，并设置合适的图片尺寸
    plt.figure(figsize=(8, 5))

    # 绘制直方图，用来观察数值变量的分布情况
    plt.hist(df[column].dropna(), bins=30, edgecolor="black", alpha=0.75)

    # 图表标题和坐标轴均使用英文，方便论文、报告或项目展示
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")

    # 自动调整布局，避免标题或坐标轴文字被裁切
    plt.tight_layout()

    # 保存图片到 figures 文件夹
    plt.savefig(FIGURES_DIR / output_file, dpi=300)

    # 关闭当前图表，释放内存，避免多张图互相影响
    plt.close()


def save_diabetes_count_plot(df):
    """绘制 diabetes=0 和 diabetes=1 的数量柱状图。"""
    if "diabetes" not in df.columns:
        print("Column not found, skip plot: diabetes")
        return

    # 统计 diabetes 每个类别的样本数量，并按标签顺序排列
    counts = df["diabetes"].value_counts().sort_index()

    plt.figure(figsize=(6, 5))
    counts.plot(kind="bar", color=["#4C78A8", "#F58518"], edgecolor="black")

    # 图表标题和坐标轴使用英文
    plt.title("Diabetes Count")
    plt.xlabel("Diabetes")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "diabetes_count.png", dpi=300)
    plt.close()


def save_relationship_plot(df, feature, output_file):
    """绘制二分类特征与 diabetes 之间关系的分组柱状图。"""
    required_columns = {feature, "diabetes"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Column not found, skip plot: {', '.join(sorted(missing_columns))}")
        return

    # 使用交叉表统计 feature 不同取值下 diabetes=0/1 的数量
    relation_table = pd.crosstab(df[feature], df["diabetes"])

    # 绘制分组柱状图，便于比较该特征与糖尿病标签之间的关系
    ax = relation_table.plot(
        kind="bar",
        figsize=(7, 5),
        edgecolor="black",
        color=["#4C78A8", "#F58518"],
    )

    # 图表标题和坐标轴使用英文
    pretty_name = feature.replace("_", " ").title()
    ax.set_title(f"{pretty_name} vs Diabetes")
    ax.set_xlabel(pretty_name)
    ax.set_ylabel("Count")
    ax.legend(title="Diabetes")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_file, dpi=300)
    plt.close()


def main():
    """执行探索性数据分析并保存图表。"""
    # 先检查数据文件是否存在；如果不存在，给出提示并结束程序
    if not CSV_PATH.exists():
        print(f"File not found: {CSV_PATH}")
        return

    # 创建 figures 文件夹；exist_ok=True 表示文件夹已存在时不会报错
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 读取 CSV 数据文件
    df = pd.read_csv(CSV_PATH)

    # 打印基础探索信息
    print_basic_info(df)

    # 绘制 age 分布图
    save_histogram(
        df,
        column="age",
        title="Age Distribution",
        xlabel="Age",
        output_file="age_distribution.png",
    )

    # 绘制 bmi 分布图
    save_histogram(
        df,
        column="bmi",
        title="BMI Distribution",
        xlabel="BMI",
        output_file="bmi_distribution.png",
    )

    # 绘制 HbA1c_level 分布图
    save_histogram(
        df,
        column="HbA1c_level",
        title="HbA1c Level Distribution",
        xlabel="HbA1c Level",
        output_file="hba1c_distribution.png",
    )

    # 绘制 blood_glucose_level 分布图
    save_histogram(
        df,
        column="blood_glucose_level",
        title="Blood Glucose Level Distribution",
        xlabel="Blood Glucose Level",
        output_file="blood_glucose_distribution.png",
    )

    # 绘制 diabetes=0 和 diabetes=1 的数量柱状图
    save_diabetes_count_plot(df)

    # 绘制 hypertension 与 diabetes 的关系图
    save_relationship_plot(
        df,
        feature="hypertension",
        output_file="hypertension_diabetes_relationship.png",
    )

    # 绘制 heart_disease 与 diabetes 的关系图
    save_relationship_plot(
        df,
        feature="heart_disease",
        output_file="heart_disease_diabetes_relationship.png",
    )

    print(f"\nEDA figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
