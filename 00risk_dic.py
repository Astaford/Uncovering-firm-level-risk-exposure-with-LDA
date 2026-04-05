# coding: gbk
from openai import OpenAI
import pandas as pd
from collections import defaultdict
import json
import time
import os
import re

# 设置环境变量
os.environ['PYTHONIOENCODING'] = 'gbk'

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key="sk-ba7e580356e447f086f01dd22491070c",
    base_url="https://api.deepseek.com"
)

# 文件路径配置
INPUT_CSV = r"C:\Users\ASUS\Desktop\test\lda_for_paper\AllData\Data2013.csv"
OUTPUT_DIR = r"C:\Users\ASUS\Desktop\test\lda_for_paper\00RiskDictionary2013"


def load_and_preprocess_data(file_path):
    """加载CSV数据并预处理"""
    try:
        df = pd.read_csv(file_path, encoding='gbk')
        print(f"成功加载数据，共 {len(df)} 行")

        # 检查必要列是否存在
        required_cols = ['Industry', 'Summary']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV文件中缺少必要列: {missing_cols}")

        # 按行业分组合并文本
        industry_data = defaultdict(str)
        industry_counts = defaultdict(int)

        for _, row in df.iterrows():
            if pd.notna(row['Industry']) and pd.notna(row['Summary']):
                industry = str(row['Industry']).strip()
                summary = str(row['Summary']).strip()
                if industry and summary:
                    industry_data[industry] += summary + "\n\n"
                    industry_counts[industry] += 1

        print(f"处理了 {len(industry_data)} 个行业")
        for industry, count in industry_counts.items():
            print(f"  - {industry}: {count} 条记录")

        return dict(industry_data)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None


def generate_industry_risks(api_client, industry_name, industry_text):
    """为每个行业生成风险主题（JSON格式）"""
    # 限制文本长度，避免超过token限制
    text_sample = industry_text[:3000]  # 进一步减少文本长度

    prompt = f"""你是一位资深行业风险分析师。
请为「{industry_name}」行业分析文本内容，提取5个最具代表性的风险主题。

要求：
1. 每个风险主题包含以下字段：
   - 风险术语（简短明确的名称）
   - 定义（详细说明）
   - 风险类型（选择：市场风险/信用风险/操作风险/法律风险/战略风险/声誉风险）
   - 严重程度（选择：高/中/低）
   - 领域分类（如：财务/运营/技术/人力资源等）

2. 按以下JSON格式输出：
{{
  "风险主题": [
    {{
      "风险术语": "风险名称",
      "定义": "风险详细定义",
      "风险类型": "具体风险类型",
      "严重程度": "高/中/低",
      "领域分类": "具体领域"
    }}
  ]
}}

行业文本内容：
{text_sample}"""

    try:
        print(f"正在为 {industry_name} 生成风险主题...")
        response = api_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的风险分析师，必须严格按照JSON格式输出结果。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        result_text = response.choices[0].message.content
        print(f"API原始响应: {result_text[:200]}...")  # 打印部分响应用于调试

        result_data = json.loads(result_text)

        # 验证响应结构
        if '风险主题' in result_data and isinstance(result_data['风险主题'], list):
            print(f"? 成功生成 {industry_name} 行业风险主题，共 {len(result_data['风险主题'])} 个主题")
            return result_data
        else:
            print(f"? {industry_name} API响应格式不符合预期")
            print(f"响应内容: {result_data}")
            return None

    except json.JSONDecodeError as e:
        print(f"? {industry_name} JSON解析失败: {e}")
        print(f"原始响应: {result_text}")
        return None
    except Exception as e:
        print(f"? 生成 {industry_name} 行业风险失败: {e}")
        return None


def generate_stopwords(api_client, risk_term, industry_name):
    """为每个风险主题生成停用词"""
    prompt = f"""请为 {industry_name} 行业的「{risk_term}」风险生成15个需要过滤的停用词。

要求：
1. 包含与该风险无关但可能同时出现的无效词汇
2. 包含可能造成干扰的近义词或相关词
3. 用中文逗号分隔，不要编号或任何解释

直接输出停用词列表，用逗号分隔："""

    try:
        response = api_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的文本分析师"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )

        text = response.choices[0].message.content.strip()
        # 清理响应文本，移除可能的说明文字
        lines = text.split('\n')
        stopwords_line = lines[-1]  # 通常最后一行是停用词列表

        # 提取逗号分隔的停用词
        stopwords = [w.strip() for w in stopwords_line.split('，') if w.strip()]
        stopwords = stopwords[:15]  # 限制数量

        print(f"  为风险 '{risk_term}' 生成 {len(stopwords)} 个停用词")
        return stopwords

    except Exception as e:
        print(f"生成停用词失败 ({risk_term}): {e}")
        return []


def process_industry(api_client, industry_name, industry_text):
    """分析单个行业"""
    print(f"\n{'=' * 50}")
    print(f"正在处理行业: {industry_name}")
    print(f"行业文本长度: {len(industry_text)} 字符")

    risk_result = generate_industry_risks(api_client, industry_name, industry_text)
    if not risk_result:
        print(f"? {industry_name} 行业风险分析失败")
        return None

    # 为每个风险主题生成停用词
    print(f"为 {industry_name} 生成停用词...")
    for risk_item in risk_result['风险主题']:
        risk_term = risk_item['风险术语']
        stopwords = generate_stopwords(api_client, risk_term, industry_name)
        risk_item['停用词'] = stopwords
        time.sleep(1)  # 避免API限制

    print(f"? 完成 {industry_name} 行业分析")
    return risk_result


def save_results(results, industry_name, output_dir):
    """保存分析结果为JSON"""
    os.makedirs(output_dir, exist_ok=True)

    # 清理文件名中的非法字符
    safe_industry_name = re.sub(r'[<>:"/\\|?*]', '_', industry_name)
    output_path = os.path.join(output_dir, f"{safe_industry_name}_风险分析.json")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"? 结果已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"? 保存 {industry_name} 结果失败: {e}")
        return False


def main():
    """主函数"""
    print("开始风险词典生成程序...")

    # 加载数据
    industry_data = load_and_preprocess_data(INPUT_CSV)
    if not industry_data:
        print("无法加载数据，程序结束。")
        return

    success_count = 0
    total_industries = len(industry_data)

    for industry_name, industry_text in industry_data.items():
        print(f"\n进度: {list(industry_data.keys()).index(industry_name) + 1}/{total_industries}")

        result = process_industry(client, industry_name, industry_text)
        if result:
            if save_results(result, industry_name, OUTPUT_DIR):
                success_count += 1

        # 增加延迟，避免API限制
        time.sleep(2)

    print(f"\n{'=' * 50}")
    print(f"程序执行完成！")
    print(f"成功处理: {success_count}/{total_industries} 个行业")
    print(f"结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()