# coding: utf-8
import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime


def process_risk_data():
    # 创建输出文件夹
    output_dir = "04monthly_risktype_score"
    os.makedirs(output_dir, exist_ok=True)

    # 读取JSON文件
    with open('03risk_classification/risk_types_supervised24.json', 'r', encoding='utf-8') as f:
        risk_data = json.load(f)

    # 构建标签到风险类别的映射
    label_to_category = {}
    for category, labels in risk_data['classified'].items():
        for label in labels:
            label_to_category[label] = category

    print(f"已加载 {len(label_to_category)} 个风险标签到类别的映射")

    # 读取所有matrix开头的CSV文件
    input_dir = "02industry_matrix24"
    file_pattern = os.path.join(input_dir, "matrix*.csv")
    csv_files = glob.glob(file_pattern)

    if not csv_files:
        print(f"在文件夹 {input_dir} 中未找到matrix开头的CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 存储所有月份的风险数据
    all_monthly_data = []

    for file_path in csv_files:
        try:
            print(f"正在处理文件: {os.path.basename(file_path)}")

            # 读取CSV文件
            df = pd.read_csv(file_path, encoding='utf-8')

            # 检查必要的列是否存在
            if 'DeclareDate' not in df.columns:
                print(f"文件 {file_path} 缺少DeclareDate列，跳过")
                continue

            # 处理日期列
            df['DeclareDate'] = pd.to_datetime(df['DeclareDate'], format='%Y/%m/%d', errors='coerce')
            df = df.dropna(subset=['DeclareDate'])

            if df.empty:
                print(f"文件 {file_path} 没有有效的日期数据，跳过")
                continue

            # 添加年月列（字符串格式，避免Period类型问题）
            df['YearMonth'] = df['DeclareDate'].dt.strftime('%Y-%m')

            # 获取Symbol列（假设是第一列）
            symbol_col = df.columns[0]

            # 识别风险标签列（第4列及以后）
            # 跳过前3列：Symbol, DeclareDate, 和其他可能的列
            label_columns = []
            for col in df.columns[3:]:
                # 检查该列是否为数值型（概率数据）
                if pd.api.types.is_numeric_dtype(df[col]):
                    label_columns.append(col)

            print(f"  识别到 {len(label_columns)} 个风险标签列")

            # 处理每个月份
            for month in df['YearMonth'].unique():
                month_data = df[df['YearMonth'] == month]

                # 处理每个公司（Symbol）
                for symbol in month_data[symbol_col].unique():
                    symbol_month_data = month_data[month_data[symbol_col] == symbol]

                    # 为每个风险类别计算平均概率
                    risk_category_scores = {}

                    # 处理每个风险标签
                    for label_col in label_columns:
                        # 计算该标签在该月份该公司的平均概率
                        avg_prob = symbol_month_data[label_col].mean()

                        if pd.isna(avg_prob):
                            continue

                        # 查找对应的风险类别
                        risk_category = label_to_category.get(label_col, '未知类别')

                        # 累加到对应的风险类别
                        if risk_category not in risk_category_scores:
                            risk_category_scores[risk_category] = []
                        risk_category_scores[risk_category].append(avg_prob)

                    # 计算每个风险类别的平均得分
                    final_scores = {}
                    for category, scores in risk_category_scores.items():
                        final_scores[category] = np.mean(scores)

                    # 创建结果行
                    result_row = {
                        'Symbol': symbol,
                        'DeclareDate': f"{month}-01",  # 使用月份第一天作为日期
                        **final_scores
                    }

                    all_monthly_data.append(result_row)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            continue

    if not all_monthly_data:
        print("没有生成任何有效数据")
        return

    # 创建最终数据框
    final_df = pd.DataFrame(all_monthly_data)

    # 填充NaN值为0
    final_df = final_df.fillna(0)

    # 确保所有风险类别列都存在
    all_categories = list(risk_data['classified'].keys()) + ['未知类别']
    for category in all_categories:
        if category not in final_df.columns:
            final_df[category] = 0

    # 按日期和Symbol排序
    final_df['SortDate'] = pd.to_datetime(final_df['DeclareDate'])
    final_df = final_df.sort_values(['SortDate', 'Symbol'])
    final_df = final_df.drop('SortDate', axis=1)

    # 确保列的顺序：Symbol, DeclareDate, 然后是各个风险类别
    base_columns = ['Symbol', 'DeclareDate']
    risk_columns = [col for col in final_df.columns if col not in base_columns]

    # 按风险类别名称排序
    risk_columns_sorted = sorted(risk_columns)
    columns_order = base_columns + risk_columns_sorted
    final_df = final_df[columns_order]

    # 保存结果
    output_file = os.path.join(output_dir, "monthly_risktype_score24.csv")
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n处理完成！结果已保存到: {output_file}")
    print(f"生成的CSV包含 {len(final_df)} 行数据")
    print(f"风险类别数量: {len(risk_columns_sorted)}")
    print("风险类别包括:", risk_columns_sorted)

    # 显示数据统计信息
    print(f"\n数据统计:")
    print(f"时间范围: {final_df['DeclareDate'].min()} 到 {final_df['DeclareDate'].max()}")
    print(f"公司数量: {final_df['Symbol'].nunique()}")
    print(f"月份数量: {final_df['DeclareDate'].nunique()}")

    # 显示每个风险类别的统计
    print(f"\n各风险类别得分统计:")
    for category in risk_columns_sorted:
        if category in final_df.columns:
            avg_score = final_df[category].mean()
            max_score = final_df[category].max()
            print(f"  {category}: 平均得分 {avg_score:.4f}, 最高得分 {max_score:.4f}")


# 运行处理函数
if __name__ == "__main__":
    process_risk_data()