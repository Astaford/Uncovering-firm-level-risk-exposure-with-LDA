# coding: gbk
import json
import os
from collections import defaultdict


def extract_industry_name_from_filename(filename):
    """从文件名提取行业名称"""
    return filename.replace("_风险分析.json", "")


def find_risk_topics(data, industry_name):
    """
    在各种可能的数据结构中查找风险主题
    """
    # 方法1: 直接查找"风险主题"键
    if "风险主题" in data:
        return "风险主题", data["风险主题"]

    # 方法2: 查找包含"风险"的键
    for key in data.keys():
        if '风险' in key:
            return key, data[key]

    # 方法3: 如果数据本身就是列表（风险主题数组）
    if isinstance(data, list):
        return "风险术语", data

    # 方法4: 查找第一个字典或列表类型的值
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            return key, value

    return None, None


def extract_risk_topics(risk_data):
    """
    从风险数据结构中提取风险主题
    """
    topics = {}

    if isinstance(risk_data, list):
        # 处理列表格式：直接是风险主题数组
        for i, item in enumerate(risk_data):
            if isinstance(item, dict) and "风险术语" in item:
                # 使用风险术语作为键，整个字典作为值
                risk_term = item["风险术语"]
                topics[risk_term] = item
            elif isinstance(item, dict):
                # 如果没有风险术语字段，使用索引作为键
                topics[f"风险主题_{i}"] = item
    elif isinstance(risk_data, dict):
        # 处理字典格式：键是风险主题名，值是风险信息
        for topic_name, topic_info in risk_data.items():
            if isinstance(topic_info, dict):
                topics[topic_name] = topic_info
    else:
        print(f"  未知的数据类型: {type(risk_data)}")

    return topics


def merge_industry_risk_dictionaries(input_dir, output_file):
    """
    合并所有行业的风险字典到一个文件中
    """
    merged_data = {
        "行业风险字典": defaultdict(dict),
        "全局风险统计": defaultdict(int),
        "文件处理统计": {
            "总文件数": 0,
            "成功处理": 0,
            "处理失败": 0
        }
    }

    processed_files = 0
    failed_files = []

    # 获取所有JSON文件
    json_files = [f for f in os.listdir(input_dir) if f.endswith("_风险分析.json")]
    merged_data["文件处理统计"]["总文件数"] = len(json_files)

    print(f"找到 {len(json_files)} 个行业文件")

    for filename in json_files:
        try:
            industry_name = extract_industry_name_from_filename(filename)
            filepath = os.path.join(input_dir, filename)

            print(f"\n处理文件: {filename}")
            print(f"行业名称: {industry_name}")

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"文件结构类型: {type(data)}")
            if isinstance(data, dict):
                print(f"字典键: {list(data.keys())}")
            elif isinstance(data, list):
                print(f"列表长度: {len(data)}")

            # 查找风险主题数据
            found_key, risk_data = find_risk_topics(data, industry_name)

            if risk_data is None:
                print(f"无法识别风险数据结构，跳过")
                failed_files.append(filename)
                continue

            print(f"找到数据键: {found_key}, 数据类型: {type(risk_data)}")

            # 提取风险主题
            risk_topics = extract_risk_topics(risk_data)

            if not risk_topics:
                print(f"未提取到风险主题，跳过")
                failed_files.append(filename)
                continue

            print(f"提取到风险主题数量: {len(risk_topics)}")

            # 显示前几个风险主题示例
            sample_keys = list(risk_topics.keys())[:3]
            print(f"  风险主题示例: {sample_keys}")

            # 合并到主字典
            merged_data["行业风险字典"][industry_name] = risk_topics

            # 统计风险类型
            risk_type_count = defaultdict(int)
            valid_topics = 0

            for topic_name, topic_data in risk_topics.items():
                if isinstance(topic_data, dict):
                    # 适配不同的字段名
                    risk_type = topic_data.get("风险类型") or \
                                topic_data.get("类型") or \
                                topic_data.get("risk_type", "其他风险")
                    risk_type_count[risk_type] += 1
                    valid_topics += 1

            # 更新全局统计
            for risk_type, count in risk_type_count.items():
                merged_data["全局风险统计"][risk_type] += count

            print(f"  有效主题: {valid_topics}/{len(risk_topics)}")
            processed_files += 1
            merged_data["文件处理统计"]["成功处理"] += 1

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_files.append(filename)
            merged_data["文件处理统计"]["处理失败"] += 1

    # 转换数据结构
    merged_data["行业风险字典"] = dict(merged_data["行业风险字典"])
    merged_data["全局风险统计"] = dict(merged_data["全局风险统计"])

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    # 输出统计
    print(f"\n" + "=" * 50)
    print(f"合并完成")
    print(f"=" * 50)
    print(f"总文件数: {len(json_files)}")
    print(f"成功处理: {processed_files} 个文件")
    print(f"失败文件: {len(failed_files)} 个")

    if failed_files:
        print("失败文件列表:")
        for f in failed_files:
            print(f"  - {f}")

    print(f"\n合并的行业数量: {len(merged_data['行业风险字典'])}")
    print("行业列表:")
    for industry in merged_data['行业风险字典'].keys():
        print(f"  - {industry}")

    print(f"\n全局风险统计:")
    total_risks = sum(merged_data['全局风险统计'].values())
    for risk_type, count in merged_data['全局风险统计'].items():
        percentage = (count / total_risks) * 100 if total_risks > 0 else 0
        print(f"  {risk_type}: {count} ({percentage:.1f}%)")

    print(f"\n输出文件: {output_file}")

    return merged_data


# 使用示例
if __name__ == "__main__":
    INPUT_DIR = r"C:\Users\ASUS\Desktop\test\lda_for_paper\00RiskDictionary2014"
    OUTPUT_FILE = r"C:\Users\ASUS\Desktop\test\lda_for_paper\00industry_risk_dics\industry_risk_dic2014.json"

    # 检查输入目录是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录不存在: {INPUT_DIR}")
    else:
        print(f"输入目录: {INPUT_DIR}")
        print(f"输出文件: {OUTPUT_FILE}")
        merge_industry_risk_dictionaries(INPUT_DIR, OUTPUT_FILE)