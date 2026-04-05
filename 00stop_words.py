from openai import OpenAI
import pandas as pd
import json
from collections import defaultdict
import time
import os
import glob

client = OpenAI(
    api_key="sk-f37260ab4c934106908d3fce64149c8e",
    base_url="https://api.deepseek.com"
)


def generate_stopwords(api_client, term_dict):
    """为每个风险术语生成30个相关停用词"""
    stopwords_dict = {}

    for term, metadata in term_dict.items():
        print(f"正在生成 [{term}] 的停用词...")

        prompt = f"""请为金融风险术语 '{term}' 生成30个需要过滤的停用词：
1. 包含该术语的常见无效变体
2. 包含相关但无分析价值的词汇
3. 包含可能造成干扰的近义词
4. 只返回纯文本，用中文逗号分隔

示例输出：
干扰词1,干扰词2,...,等30个

请直接返回30个用逗号分隔的词语，不要任何解释："""

        try:
            response = api_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是专业的金融文本处理专家"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=200,
                stream=False
            )

            # 解析返回的停用词
            raw_text = response.choices[0].message.content
            stopwords = [word.strip() for word in raw_text.split('，') if word.strip()][:30]  # 确保30个
            stopwords_dict[term] = stopwords

            # API限速控制
            time.sleep(1)

        except Exception as e:
            print(f"生成 [{term}] 停用词失败: {e}")
            stopwords_dict[term] = []

    return stopwords_dict


def process_industry_file(input_file_path, output_folder):
    """处理单个行业文件"""
    try:
        # 从文件名提取行业名称
        file_name = os.path.basename(input_file_path)
        industry_name = file_name.replace('_风险分析.json', '')

        print(f"\n正在处理行业: {industry_name}")

        # 读取风险分析文件
        with open(input_file_path, 'r', encoding='utf-8') as f:
            risk_dict = json.load(f)

        # 获取行业名称作为顶级键
        industry_key = list(risk_dict.keys())[0]  # 获取第一个键（行业名称）
        risk_topics = risk_dict[industry_key]

        # 为每个风险术语生成停用词
        stopwords_dict = generate_stopwords(client, risk_topics)

        # 合并结果
        for term in risk_topics:
            if term in stopwords_dict:
                risk_topics[term]["停用词"] = stopwords_dict[term]

        # 构建输出文件路径
        output_file_name = f"{industry_name}_风险词典.json"
        output_file_path = os.path.join(output_folder, output_file_name)

        # 保存增强版词典
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(risk_dict, f, indent=2, ensure_ascii=False)

        print(f"增强版词典已保存到 {output_file_path}")
        print(f"{industry_name} 各主题停用词数量统计:")
        for term, words in stopwords_dict.items():
            print(f"- {term}: {len(words)}个停用词")

        return True

    except Exception as e:
        print(f"处理文件 {input_file_path} 时出错: {e}")
        return False


def main():
    # 设置文件夹路径
    folder_path = r"C:\Users\ASUS\Desktop\test\lda_for_paper\RiskDictionary2021"

    # 查找所有以"_风险分析.json"结尾的文件
    pattern = os.path.join(folder_path, "*_风险分析.json")
    input_files = glob.glob(pattern)

    if not input_files:
        print(f"在文件夹 {folder_path} 中未找到符合条件的文件")
        return

    print(f"找到 {len(input_files)} 个待处理文件:")
    for file in input_files:
        print(f"- {os.path.basename(file)}")

    # 处理每个文件
    success_count = 0
    for input_file in input_files:
        if process_industry_file(input_file, folder_path):
            success_count += 1

    print(f"\n处理完成! 成功处理 {success_count}/{len(input_files)} 个文件")


if __name__ == "__main__":
    main()