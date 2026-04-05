# coding: utf-8
import pandas as pd
import os
import re
import glob


def process_all_industry_files():
    # 行业名和编号对应表
    industry_list = {
        "农、林、牧、渔业": "01",
        "采矿业": "02",
        "制造业": "16",
        "电力、热力、燃气及水生产和供应业": "03",
        "建筑业": "05",
        "交通运输、仓储和邮政业": "06",
        "信息传输、软件和信息技术服务业": "15",
        "金融业": "08",
        "房地产": "04",
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

    # 输入文件夹 - 使用相对路径或绝对路径
    input_folder = "01industry_outputs14"
    input_files = glob.glob(os.path.join(input_folder, "topics_*.csv"))

    if not input_files:
        print(f"在文件夹 '{input_folder}' 中没有找到 topics_*.csv 文件")
        print(f"该文件夹中的文件:")
        for item in os.listdir(input_folder):
            print(f"  - {item}")
        return

    print(f"找到 {len(input_files)} 个文件需要处理:")
    for file in input_files:
        print(f"  - {file}")

    # 处理每个文件
    for input_file in input_files:
        try:
            # 从文件名中提取行业名
            filename = os.path.basename(input_file)
            # 去掉"topics_"前缀和".csv"后缀，得到行业名
            industry_name = filename[7:-4]  # 去掉"topics_"和".csv"

            # 获取对应的编号
            if industry_name in industry_list:
                number = industry_list[industry_name]
                output_filename = f"topic_label{number}.csv"
            else:
                print(f"警告: 文件 {filename} 中的行业名 '{industry_name}' 不在行业列表中")
                # 如果没有找到对应的行业，使用原文件名
                output_filename = f"topic_label_{industry_name}.csv"

            # 处理单个文件
            process_single_file(input_file, output_filename, industry_name)

        except Exception as e:
            print(f"处理文件 {input_file} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n所有文件处理完成！")


def process_single_file(input_file, output_filename, industry_name):
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)

        # 检查是否包含必要的列
        if 'Risk Label' not in df.columns:
            print(f"错误: 文件 {input_file} 中没有 'Risk Label' 列")
            print(f"文件中的列: {list(df.columns)}")
            return

        # 提取所有唯一的风险标签
        all_risk_labels = set()

        # 遍历Risk Label列，提取所有标签名称
        for risk_label_str in df['Risk Label']:
            if pd.isna(risk_label_str):
                continue

            # 使用正则表达式提取标签名称（括号前的部分）
            labels = re.findall(r'([^()]+)\s*\(\d+\.\d+\)', str(risk_label_str))
            all_risk_labels.update([label.strip() for label in labels])

        # 将集合转换为排序后的列表
        all_risk_labels = sorted(list(all_risk_labels))

        # 为每个风险标签创建新列，初始值为0
        for label in all_risk_labels:
            df[label] = 0.0

        # 填充风险标签的概率值
        for idx, row in df.iterrows():
            risk_label_str = row['Risk Label']
            if pd.isna(risk_label_str):
                continue

            # 提取每个标签和对应的概率
            matches = re.findall(r'([^()]+)\s*\((\d+\.\d+)\)', str(risk_label_str))

            for label, prob in matches:
                label = label.strip()
                df.at[idx, label] = float(prob)

        # 创建输出文件夹（如果不存在）
        output_dir = '01topic_labels14'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存处理后的文件
        output_path = os.path.join(output_dir, output_filename)
        df.to_csv(output_path, index=False, encoding='gbk')

        print(f"行业 '{industry_name}' 处理完成！输出文件: {output_filename}")
        print(f"  提取到的风险标签数量: {len(all_risk_labels)}")
        print(f"  处理前的列数: {len(df.columns) - len(all_risk_labels)}, 处理后的列数: {len(df.columns)}")

    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()


# 运行程序
if __name__ == "__main__":
    process_all_industry_files()