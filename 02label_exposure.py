import os
import pandas as pd
import numpy as np
import re

# 配置路径
TOPIC_LABELS_DIR = "topic_labels24"
INDUSTRY_OUTPUT_DIR = "industry_outputs24"
RESULT_DIR = "industry_matrix24"

# 行业名称与编号映射
industry_list = {
    "农、林、牧、渔业": "01",
    "采矿业": "02",
    "制造业": "16",
    "电力、热力、燃气及水生产和供应业": "03",
    "建筑业": "05",
    "交通运输、仓储和邮政业": "06",
    "信息传输、软件和信息技术服务业": "15",
    "金融业": "08",
    "房地产业": "04",
    "租赁和商务服务业": "19",
    "综合": "18",
    "批发和零售业": "11",
    "水利、环境和公共设施管理业": "12",
    "文化、体育和娱乐业": "14",
    "居民服务、修理和其他服务业": "09",
    "卫生和社会工作": "13",
    "教育": "07",
    "住宿和餐饮业": "17",
    "科学研究和技术服务业": "10"
}

# 创建结果目录
os.makedirs(RESULT_DIR, exist_ok=True)


def process_industry(industry_name, industry_code):
    """处理单个行业的矩阵运算"""

    # 1. 读取topic_label文件 (10×n矩阵)
    labels_file = os.path.join(TOPIC_LABELS_DIR, f"topic_label{industry_code}.csv")
    labels_df = pd.read_csv(labels_file, encoding='gbk')

    # 从第3列开始读取风险标签（跳过前2列）
    risk_label_columns = labels_df.columns[2:]  # 第3列到最后一列

    # 构建10×n的风险矩阵，确保转换为数值类型
    risk_matrix = labels_df[risk_label_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values

    # 获取所有风险标签名称
    all_risk_labels = list(risk_label_columns)

    # 2. 读取行业output文件 (m×10矩阵)
    output_file = os.path.join(INDUSTRY_OUTPUT_DIR, f"output_{industry_code}_{industry_name}.csv")
    output_df = pd.read_csv(output_file)

    # 确保有Symbol列和topic_0到topic_9列
    topic_cols = [f"topic_{i}" for i in range(10)]
    symbol_topic_matrix = output_df[topic_cols].values
    symbols = output_df['Symbol'].values
    # 获取DeclareDate列
    if 'DeclareDate' in output_df.columns:
        declare_dates = output_df['DeclareDate'].values
    else:
        print(f"警告: {industry_name} 的output文件中没有DeclareDate列")
        declare_dates = [None] * len(symbols)

    # 3. 矩阵乘法 (m×10 @ 10×n = m×n)
    result_matrix = np.dot(symbol_topic_matrix, risk_matrix)

    # 4. 保存结果
    result_df = pd.DataFrame(result_matrix,
                             index=symbols,
                             columns=all_risk_labels)
    result_df.reset_index(inplace=True)
    result_df.rename(columns={'index': 'Symbol'}, inplace=True)
    # 在Symbol列后插入DeclareDate列
    result_df.insert(1, 'DeclareDate', declare_dates)

    # 保存到CSV
    output_path = os.path.join(RESULT_DIR, f"matrix_{industry_code}_{industry_name}.csv")
    result_df.to_csv(output_path, index=False)
    print(f"已处理 {industry_name}({industry_code}) 行业，结果保存至 {output_path}")


def main():
    # 处理每个行业
    for industry_name, industry_code in industry_list.items():
        try:
            process_industry(industry_name, industry_code)
        except Exception as e:
            print(f"处理行业 {industry_name}({industry_code}) 时出错: {str(e)}")


if __name__ == "__main__":
    main()