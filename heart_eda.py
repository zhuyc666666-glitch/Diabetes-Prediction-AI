from pathlib import Path

import pandas as pd


# 心脏病数据文件路径：本脚本只做数据检查，不训练模型，也不生成任何模型文件。
DATA_PATH = Path("data") / "heart_disease_combined.csv"

# 可能作为目标列的候选名称。
TARGET_CANDIDATES = [
    "target",
    "HeartDisease",
    "num",
    "heart_disease",
    "diagnosis",
    "condition",
]


def find_target_column(columns):
    """从候选名称中自动判断数据集里的目标列。"""
    # 先做精确匹配，避免误判。
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate

    # 如果大小写不完全一致，再做一次忽略大小写的匹配。
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in TARGET_CANDIDATES:
        matched_column = lower_to_original.get(candidate.lower())
        if matched_column is not None:
            return matched_column

    return None


def main():
    """读取心脏病数据并打印基础检查信息。"""
    # 检查数据文件是否存在，避免路径错误时出现难以理解的报错。
    if not DATA_PATH.exists():
        print(f"未找到数据文件: {DATA_PATH}")
        return

    # 使用 pandas 读取 CSV 数据。
    df = pd.read_csv(DATA_PATH)

    print("=" * 80)
    print("1. 数据前5行")
    print("=" * 80)
    print(df.head())

    print("\n" + "=" * 80)
    print("2. 数据维度")
    print("=" * 80)
    print(df.shape)

    print("\n" + "=" * 80)
    print("3. 所有列名")
    print("=" * 80)
    print(list(df.columns))

    print("\n" + "=" * 80)
    print("4. 每一列的数据类型")
    print("=" * 80)
    print(df.dtypes)

    print("\n" + "=" * 80)
    print("5. 每一列缺失值数量")
    print("=" * 80)
    print(df.isnull().sum())

    print("\n" + "=" * 80)
    print("6. 每一列唯一值数量")
    print("=" * 80)
    print(df.nunique())

    print("\n" + "=" * 80)
    print("7. 自动判断可能的目标列")
    print("=" * 80)
    target_column = find_target_column(df.columns)
    if target_column is None:
        print(f"未找到目标列。候选名称包括: {TARGET_CANDIDATES}")
    else:
        print(f"检测到目标列: {target_column}")
        print("\n目标列 0 和 1 的数量:")
        print(df[target_column].value_counts().sort_index())

        # 如果目标列是 num，需要将多分类心脏病严重程度转换为二分类标签：
        # num = 0 表示无心脏病，heart_disease = 0；
        # num > 0 表示有心脏病，heart_disease = 1。
        if target_column.lower() == "num":
            df["heart_disease"] = (df[target_column] > 0).astype(int)

            print("\n已根据 num 新增二分类标签列: heart_disease")
            print("heart_disease 中 0 和 1 的数量:")
            print(df["heart_disease"].value_counts().sort_index())


if __name__ == "__main__":
    main()
