# coding: utf-8
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义类别翻译
category_translation = {
    "市场环境类": "Market Risk",
    "经营与盈利类": "Profitability Risk",
    "政策监管类": "Policy Regulation",
    "自然与外部环境类": "Natural Disasters",
    "金融与资本类": "Financial Risk",
    "成本与价格类": "Cost Risk",
    "技术研发类": "Technology Risk"
}

# 加载聚类结果
try:
    with open('03risk_classification/risk_types_supervised24.json', 'r', encoding='utf-8') as f:
        risk_data = json.load(f)
    classified = risk_data['classified']
    print("成功加载JSON分类数据")
except Exception as e:
    print(f"加载JSON文件时出错: {e}")
    exit()

# 构建标签到类别的映射字典
label_to_category = {}
for category, labels in classified.items():
    for label in labels:
        label_to_category[label] = category

print(f"构建了包含 {len(label_to_category)} 个标签的映射字典")

# 读取所有行业CSV文件
industry_folder = 'topic_labels24'
if not os.path.exists(industry_folder):
    print(f"文件夹 '{industry_folder}' 不存在")
    exit()

industry_files = [f for f in os.listdir(industry_folder) if f.startswith('topic_label') and f.endswith('.csv')]
print(f"找到 {len(industry_files)} 个行业文件")

if len(industry_files) == 0:
    print("没有找到符合条件的CSV文件")
    exit()

# 存储结果
results = {}

# 处理每个行业文件
for file in industry_files:
    try:
        file_path = os.path.join(industry_folder, file)
        print(f"处理文件: {file}")

        # 读取CSV文件
        df = pd.read_csv(file_path, header=None, encoding='gbk')
        print(f"文件形状: {df.shape}")

        # 检查文件是否有足够的数据
        if df.shape[0] < 2 or df.shape[1] < 3:
            print(f"文件 {file} 数据不足，跳过")
            continue

        # 提取风险标签（第4列开始）
        risk_labels = df.iloc[0, 3:].dropna().tolist()
        print(f"找到 {len(risk_labels)} 个风险标签")

        if len(risk_labels) == 0:
            print(f"文件 {file} 没有风险标签，跳过")
            continue

        # 提取概率（第2~11行，第3列开始）
        probabilities = df.iloc[1:11, 2:].copy()

        # 确保概率数据是数值类型
        for col in probabilities.columns:
            probabilities[col] = pd.to_numeric(probabilities[col], errors='coerce')

        probabilities = probabilities.values
        print(f"概率矩阵形状: {probabilities.shape}")

        # 按类别聚合概率
        category_probs = {}
        for i, label in enumerate(risk_labels):
            if label in label_to_category:
                category = label_to_category[label]
                # 获取该标签在所有行中的概率
                if i < probabilities.shape[1]:  # 确保索引不越界
                    label_probs = probabilities[:, i]
                    # 只取非零且非NaN的概率值计算平均值
                    valid_probs = label_probs[(label_probs > 0) & (~np.isnan(label_probs))]
                    if len(valid_probs) > 0:
                        if category not in category_probs:
                            category_probs[category] = []
                        category_probs[category].append(np.mean(valid_probs))
                        print(f"  标签 '{label}' -> 类别 '{category}': 平均概率 {np.mean(valid_probs):.4f}")

        # 计算每个类别的平均概率
        industry_result = {}
        for category in category_translation.keys():
            if category in category_probs and len(category_probs[category]) > 0:
                industry_result[category] = np.mean(category_probs[category])
            else:
                industry_result[category] = 0.0

        # 提取行业名称
        industry_name = file.replace('topic_label', '').replace('.csv', '')
        results[industry_name] = industry_result
        print(f"行业 {industry_name} 处理完成")

    except Exception as e:
        print(f"处理文件 {file} 时出错: {e}")
        continue

if len(results) == 0:
    print("没有成功处理任何文件")
    exit()

# 转换为DataFrame
df_result = pd.DataFrame(results).T
df_result = df_result.reindex(columns=category_translation.keys())

# 重命名列为英文
df_result.columns = [category_translation[col] for col in df_result.columns]

print(f"结果表格形状: {df_result.shape}")
print("前几行数据:")
print(df_result.head())

# 保存为CSV
output_file = 'industry_risk_type/industry_risk_probabilities24.csv'
df_result.to_csv(output_file, float_format='%.4f')
print(f"结果已保存到: {output_file}")

# 可视化表格
plt.figure(figsize=(16, 10))

# 创建表格数据
cell_text = np.round(df_result.values, 4)

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

# 创建表格
table = ax.table(
    cellText=cell_text,
    rowLabels=df_result.index,
    colLabels=df_result.columns,
    cellLoc='center',
    loc='center'
)

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# 设置表格颜色 - 修复索引问题
n_rows, n_cols = len(df_result) + 1, len(df_result.columns)
for i in range(n_rows):
    for j in range(n_cols):
        if i == 0:  # 表头行
            table[(i, j)].set_facecolor('#4CAF50')
            table[(i, j)].set_text_props(weight='bold', color='white')
        elif i % 2 == 1:  # 奇数行
            table[(i, j)].set_facecolor('#f5f5f5')
        else:  # 偶数行
            table[(i, j)].set_facecolor('white')

plt.title('Industry Risk Probability 2024', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# 创建热图可视化
plt.figure(figsize=(12, 8))
im = plt.imshow(df_result.values, cmap='YlOrRd', aspect='auto')

# 设置坐标轴
plt.xticks(range(len(df_result.columns)), df_result.columns, rotation=45, ha='right')
plt.yticks(range(len(df_result.index)), df_result.index)

# 添加颜色条
plt.colorbar(im, label='Risk Probability')

# 添加数值标注
for i in range(len(df_result.index)):
    for j in range(len(df_result.columns)):
        text = plt.text(j, i, f'{df_result.values[i, j]:.3f}',
                        ha="center", va="center", color="black", fontsize=8)

plt.title('Industry Risk Probability 2024', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# 打印统计信息
print("\n统计信息:")
print(f"处理的行业数量: {len(results)}")
print(f"风险类别数量: {len(category_translation)}")

# 打印每个行业的最高风险类别
print("\n各行业最高风险类别:")
for industry, row in df_result.iterrows():
    max_risk = row.idxmax()
    max_value = row.max()
    print(f"{industry}: {max_risk} ({max_value:.4f})")