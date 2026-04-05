# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from datetime import datetime
import matplotlib as mpl

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 风险类别翻译
category_translation = {
    "市场环境类": "Market Risk",
    "经营与盈利类": "Profitability Risk",
    "政策监管类": "Policy Regulation",
    "自然与外部环境类": "Natural Disasters",
    "金融与资本类": "Financial Risk",
    "成本与价格类": "Cost Risk",
    "技术研发类": "Technology Risk"
}

def create_risk_trend_plots():
    # 读取所有CSV文件
    input_dir = "04monthly_risktype_score"
    file_pattern = os.path.join(input_dir, "monthly_risktype_score*.csv")
    csv_files = glob.glob(file_pattern)

    if not csv_files:
        print(f"在文件夹 {input_dir} 中未找到monthly_risktype_score*.csv文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 合并所有数据
    all_data = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
            all_data.append(df)
            print(f"已读取: {os.path.basename(file_path)} - {len(df)} 行数据")
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")

    if not all_data:
        print("没有成功读取任何数据")
        return

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)

    # 处理日期列
    combined_df['Date'] = pd.to_datetime(combined_df['DeclareDate'])
    combined_df['Year'] = combined_df['Date'].dt.year
    combined_df['Month'] = combined_df['Date'].dt.month

    # 筛选2015-2024年的数据
    combined_df = combined_df[(combined_df['Year'] >= 2015) & (combined_df['Year'] <= 2024)]

    if combined_df.empty:
        print("没有找到2015-2024年的数据")
        return

    print(f"合并后数据量: {len(combined_df)} 行")
    print(f"年份范围: {combined_df['Year'].min()} - {combined_df['Year'].max()}")

    # 创建输出文件夹
    output_dir = "risk_trend_plots"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 总体风险趋势图
    create_overall_risk_trend(combined_df, output_dir)

    # 2. 各类风险趋势图
    create_individual_risk_trends(combined_df, output_dir)

    print("所有图表已生成完成！")


def create_overall_risk_trend(df, output_dir):
    """创建总体风险趋势图 - 10条曲线对应10个年份"""
    print("正在生成总体风险趋势图...")

    # 计算每月所有风险类别的平均值
    monthly_avg = df.groupby(['Year', 'Month']).mean(numeric_only=True).reset_index()

    # 计算总体风险得分（所有风险类别的平均值）
    risk_columns = [col for col in df.columns if col in category_translation.keys()]
    monthly_avg['Overall_Risk'] = monthly_avg[risk_columns].mean(axis=1)

    # 创建图表
    plt.figure(figsize=(15, 8))

    # 为每个年份创建数据序列
    years = sorted(df['Year'].unique())

    # 使用tab10颜色映射，确保有10种不同颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(years)))

    # 检查是否有完整的10年数据
    if len(years) == 10:
        print("找到完整的10年数据(2015-2024)")
    else:
        print(f"找到 {len(years)} 年数据: {years}")

    for i, year in enumerate(years):
        year_data = monthly_avg[monthly_avg['Year'] == year]
        if len(year_data) > 0:
            # 确保每个月都有数据点，如果没有则用NaN填充
            complete_months = pd.DataFrame({'Month': range(1, 13)})
            year_complete = complete_months.merge(year_data, on='Month', how='left')

            plt.plot(year_complete['Month'], year_complete['Overall_Risk'],
                     marker='o', linewidth=2.5, markersize=6,
                     color=colors[i], label=f'{year}', alpha=0.8)

    plt.xlabel('Month', fontsize=14, fontweight='bold')
    plt.ylabel('Overall Risk Score', fontsize=14, fontweight='bold')
    plt.title('Overall Risk Trend (2015-2024)', fontsize=16, fontweight='bold')

    # 图例放在图表外部右侧
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
               frameon=True, fancybox=True, shadow=True, ncol=1)

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(range(1, 13), [f'{i}' for i in range(1, 13)])
    plt.xlim(0.5, 12.5)

    # 设置y轴范围，确保所有数据可见
    y_min = monthly_avg['Overall_Risk'].min() * 0.9
    y_max = monthly_avg['Overall_Risk'].max() * 1.1
    plt.ylim(y_min, y_max)

    # 调整布局，为图例留出空间
    plt.tight_layout()

    # 保存图片
    output_file = os.path.join(output_dir, "overall_risk_trend.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"总体风险趋势图已保存: {output_file}")


def create_individual_risk_trends(df, output_dir):
    """创建各类风险趋势图 - 每个风险类别一张图，每张图包含10个年份的曲线"""
    print("正在生成各类风险趋势图...")

    # 风险类别列表（使用原始中文名称）
    risk_categories = [col for col in df.columns if col in category_translation.keys()]

    # 计算每月各风险类别的平均值
    monthly_avg = df.groupby(['Year', 'Month']).mean(numeric_only=True).reset_index()

    years = sorted(df['Year'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(years)))

    # 为每个风险类别创建图表
    for category in risk_categories:
        english_name = category_translation.get(category, category)

        print(f"  正在处理: {category} -> {english_name}")

        # 创建图表
        plt.figure(figsize=(15, 8))

        # 为每个年份创建数据序列
        for i, year in enumerate(years):
            year_data = monthly_avg[monthly_avg['Year'] == year]
            if len(year_data) > 0 and category in year_data.columns:
                # 确保每个月都有数据点
                complete_months = pd.DataFrame({'Month': range(1, 13)})
                year_complete = complete_months.merge(year_data, on='Month', how='left')

                plt.plot(year_complete['Month'], year_complete[category],
                         marker='s', linewidth=2, markersize=5,
                         color=colors[i], label=f'{year}', alpha=0.8)

        plt.xlabel('Month', fontsize=14, fontweight='bold')
        plt.ylabel(f'{english_name} Score', fontsize=14, fontweight='bold')
        plt.title(f'{english_name} Trend (2015-2024)', fontsize=16, fontweight='bold')

        # 图例放在图表外部右侧
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                   frameon=True, fancybox=True, shadow=True, ncol=1)

        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(range(1, 13), [f'{i}' for i in range(1, 13)])
        plt.xlim(0.5, 12.5)

        # 设置y轴范围
        if category in monthly_avg.columns:
            y_min = monthly_avg[category].min() * 0.9
            y_max = monthly_avg[category].max() * 1.1
            plt.ylim(y_min, y_max)

        # 调整布局
        plt.tight_layout()

        # 保存图片
        filename = f"{english_name.lower().replace(' ', '_')}_trend.png"
        output_file = os.path.join(output_dir, filename)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    print("各类风险趋势图已全部生成完成！")


def print_data_summary(df):
    """打印数据摘要信息"""
    print("\n数据摘要信息:")
    print(f"总数据行数: {len(df)}")
    print(f"年份范围: {df['Year'].min()} - {df['Year'].max()}")
    print(f"包含的年份: {sorted(df['Year'].unique())}")
    print(f"风险类别数量: {len([col for col in df.columns if col in category_translation.keys()])}")

    # 显示每个年份的数据量
    year_counts = df['Year'].value_counts().sort_index()
    print("\n各年份数据量:")
    for year, count in year_counts.items():
        print(f"  {year}: {count} 行数据")


# 运行主函数
if __name__ == "__main__":
    create_risk_trend_plots()