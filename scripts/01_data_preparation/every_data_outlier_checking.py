import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# %%
# ==========================================
# 1. 数据加载 (使用你的代码片段)
# ==========================================
data_save_path = '/data/project/sleep_ENIGMA_Cognition/Codes/ENIGMA_Sleep_Cognitive/Data/'
# 如果你是在本地运行，记得修改这里的路径
try:
    df_SHIP = pd.read_csv(data_save_path + 'SHIP_Trend_dataset_renamed_all_cleaned.csv')
    print("✅ 数据加载成功！")
except FileNotFoundError:
    print("❌ 找不到文件，请检查路径。生成模拟数据用于演示...")
    # 生成模拟数据防止报错（仅供测试代码逻辑）
    np.random.seed(42)
    df_SHIP = pd.DataFrame({
        'Stroop_Test': np.random.normal(50, 10, 1000),
        'Memory_Test': np.random.normal(30, 5, 1000),
        'Age_at_Scan': np.random.randint(20, 80, 1000),
        'PSG_Sleep_Dur': np.random.normal(400, 50, 1000)
    })
    # 人为添加几个极端异常值
    df_SHIP.loc[0, 'Stroop_Test'] = 500  # 离谱的高分
    df_SHIP.loc[1, 'Memory_Test'] = -50  # 不可能的负分

targets = ['Stroop_Test', 'Memory_Test']
key_features = ['Age_at_Scan', 'PSG_Sleep_Dur']  # 挑选几个关键变量一起看

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# %%
# ==========================================
# 2. 定义检测函数
# ==========================================

def analyze_outliers(df, col_name):
    """
    对单变量进行全套异常值检测：统计、Z-Score、IQR
    """
    print(f"\n{'=' * 20} 正在分析变量: {col_name} {'=' * 20}")

    # 移除空值进行计算
    data = df[col_name].dropna()

    # --- A. Z-Score 方法 (假设正态分布) ---
    z_scores = np.abs(stats.zscore(data))
    outliers_z = data[z_scores > 3]
    z_ratio = len(outliers_z) / len(data)

    # --- B. IQR 方法 (箱线图逻辑，更鲁棒) ---
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]
    iqr_ratio = len(outliers_iqr) / len(data)

    # --- C. 打印报告 ---
    print(f"有效数据量: {len(data)}")
    print(f"数据范围: Min={data.min():.2f}, Max={data.max():.2f}, Mean={data.mean():.2f}")
    print(f"[Z-Score > 3] 发现异常值: {len(outliers_z)} 个 ({z_ratio:.2%})")
    print(f"[IQR 1.5倍法则] 发现异常值: {len(outliers_iqr)} 个 ({iqr_ratio:.2%})")

    if len(outliers_z) > 0:
        print(f"Z-Score 极端值示例: {outliers_z.head(3).values}")

    return outliers_iqr.index.tolist()


def plot_visualizations(df, targets, key_covariate='Age_at_Scan'):
    """
    绘制箱线图、直方图和散点图
    """
    for target in targets:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. 箱线图 (Boxplot) - 最直观的异常值展示
        sns.boxplot(x=df[target], ax=axes[0], color='skyblue')
        axes[0].set_title(f'{target} - Boxplot (Look for black dots)')

        # 2. 直方图 (Distribution) - 看分布是否长尾
        sns.histplot(df[target], kde=True, ax=axes[1], color='orange')
        axes[1].set_title(f'{target} - Distribution')

        # 3. 散点图 (Bivariate) - 看它和年龄的关系 (寻找离群点)
        if key_covariate in df.columns:
            sns.scatterplot(x=df[key_covariate], y=df[target], ax=axes[2], alpha=0.6)
            axes[2].set_title(f'{target} vs {key_covariate}')
            # 这里的孤立点通常是高杠杆点 (High Leverage Points)

        plt.tight_layout()
        plt.show()
        print(f"已生成 {target} 的可视化图表。")


def simple_residual_check(df, target, features):
    """
    使用简单的线性回归快速检查残差 (Residual Analysis)
    这能发现那些违背数据整体趋势的异常点
    """
    print(f"\n--- 正在对 {target} 进行残差分析 ---")

    # 准备数据 (简单处理：只取数值型特征且去空值)
    subset = df[features + [target]].dropna()
    if subset.empty:
        print("数据为空，跳过残差分析")
        return

    X = subset[features]
    y = subset[target]

    # 快速拟合线性模型
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    residuals = y - preds

    # 标准化残差 (Standardized Residuals) > 3 算异常
    std_residuals = np.abs(residuals / np.std(residuals))
    outlier_count = np.sum(std_residuals > 3)

    print(f"基于 {features} 的预测模型：")
    print(f"残差 > 3 sigma 的异常样本数: {outlier_count}")

    # 画残差图
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=preds, y=residuals)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {target} (Outliers are far from red line)')
    plt.show()


# %%
# ==========================================
# 3. 执行主程序
# ==========================================

print(">>> 第一步：统计方法检测 (Z-Score & IQR)")
for t in targets:
    analyze_outliers(df_SHIP, t)

print("\n>>> 第二步：可视化检测 (请查看生成的图表)")
# 传入 Age_at_Scan 作为参考变量，看看是否有年龄对应的异常值
plot_visualizations(df_SHIP, targets, key_covariate='Age_at_Scan')

print("\n>>> 第三步：模型残差检测 (Residual Analysis)")
# 我们用 Age, SEX, BMI 加上几个脑区特征做一个简单的基准预测
# 注意：你需要确保这些列在 CSV 里都存在且为数值型 (SEX可能需要编码)
numeric_cov = ['Age_at_Scan', 'BMI', 'PSG_Sleep_Dur']
available_covs = [c for c in numeric_cov if c in df_SHIP.columns]

if available_covs:
    for t in targets:
        simple_residual_check(df_SHIP, t, available_covs)
else:
    print("找不到用于残差分析的数值型协变量，跳过此步。")


