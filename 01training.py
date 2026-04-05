# coding: utf-8
import os
# 设置环境变量强制离线
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 然后导入其他库...
import pandas as pd
from gensim import corpora, models
import jieba
import re
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
# 设置环境变量



# --------------------- 加载通用停用词 ---------------------
with open('stopwords.txt', 'r', encoding='utf-8') as f:
    base_stopwords = set(line.strip() for line in f if line.strip())

# --------------------- 加载行业风险词典 ---------------------
with open('00industry_risk_dics/industry_risk_dic2014.json', 'r', encoding='utf-8') as f:
    full_dict = json.load(f)
    industry_dicts = full_dict.get("行业风险字典", {})

# --------------------- 加载中文向量模型 ---------------------
model = SentenceTransformer('shibing624/text2vec-base-chinese')

# --------------------- 文本预处理函数 ---------------------
def create_preprocess_function(risk_keywords, combined_stopwords):
    def preprocess(text):
        text = re.sub(r'[^\w\s]', '', str(text))
        words = jieba.lcut(text)

        def is_valid_token(word):
            if word in combined_stopwords or len(word) <= 1:
                return False
            if re.match(r'^\d+$', word) or re.match(r'^\d{2,4}[a-zA-Z]*$', word):
                return False
            return True

        filtered = [word for word in words if is_valid_token(word)]
        matched_risks = [word for word in filtered if word in risk_keywords]
        return filtered + matched_risks
    return preprocess

# --------------------- 行业编号与名称映射 ---------------------
industry_list = {
    "01": "农、林、牧、渔业",
    "02": "采矿业",
    "03": "电力、热力、燃气及水生产和供应业",
    "04": "房地产业",
    "05": "建筑业",
    "06": "交通运输、仓储和邮政业",
    "07": "教育",
    "08": "金融业",
    "09": "居民服务、修理和其他服务业",
    "10": "科学研究和技术服务业",
    "11": "批发和零售业",
    "12": "水利、环境和公共设施管理业",
    "13": "卫生和社会工作",
    "14": "文化、体育和娱乐业",
    "15": "信息传输、软件和信息技术服务业",
    "16": "制造业",
    "17": "住宿和餐饮业",
    "18": "综合",
    "19": "租赁和商务服务业"
}

# --------------------- 读取主数据 ---------------------
df_all = pd.read_csv('AllData/Data2014.csv', encoding='gbk')
assert 'Industry' in df_all.columns and 'Number' in df_all.columns, "数据中必须包含 'Industry' 和 'Number' 列"

# --------------------- 每个行业独立处理  ---------------------
Industry_groups = df_all.groupby('Number')
top_n = 3
min_score = 0.4
num_topics = 10

for number, group_df in Industry_groups:
    industry_name = industry_list.get(str(number).zfill(2))
    if not industry_name:
        print(f"编号 {number} 无对应行业名称，跳过。")
        continue

    print(f"\n处理行业：{industry_name}（编号 {number}），样本数：{len(group_df)}")

    # 行业风险词处理
    if industry_name in industry_dicts:
        risk_items = industry_dicts[industry_name]
        raw_terms = {}
        risk_keywords = set()
        for label, item in risk_items.items():
            stopword_entries = item.get('停用词', [])
            all_words = set()
            for entry in stopword_entries:
                all_words.update(w.strip() for w in entry.split(',') if w.strip())
            if all_words:
                raw_terms[label] = {'停用词': list(all_words)}
                risk_keywords.update(all_words)
        print(f"加载行业风险词典：{industry_name}（共 {len(risk_keywords)} 个风险词）")
    else:
        print(f"行业 {industry_name} 未在风险词典中找到，将仅使用通用停用词。")
        raw_terms = {}
        risk_keywords = set()

    preprocess = create_preprocess_function(risk_keywords, base_stopwords)

    # 文本分词
    group_df = group_df.copy()
    group_df['tokens'] = group_df['Summary'].apply(preprocess)

    # --------------------- 构建LDA模型 ---------------------
    dictionary = corpora.Dictionary(group_df['tokens'])
    corpus = [dictionary.doc2bow(tokens) for tokens in group_df['tokens']]
    dictionary.save('lda_dictionary.dict')
    lda_model = models.LdaModel(
        corpus=corpus,
        num_topics=num_topics,
        id2word=dictionary,
        random_state=100,
        passes=10
    )
    lda_model.save('lda_model.model')
    print("已保存LDA模型")

    # --------------------- 文档-主题分布 ---------------------
    doc_topics = [lda_model[doc] for doc in corpus]
    topic_prob_matrix = np.zeros((len(group_df), num_topics))
    for i, doc in enumerate(doc_topics):
        for topic_id, prob in doc:
            topic_prob_matrix[i, topic_id] = prob

    for i in range(num_topics):
        group_df[f'topic_{i}'] = topic_prob_matrix[:, i]

    group_df['dominant_topic'] = [
        max(doc, key=lambda x: x[1])[0] if doc else None for doc in doc_topics
    ]

    # --------------------- 生成主题关键词 ---------------------
    topic_texts = []
    topic_keywords_map = {}
    for topic_id in range(num_topics):
        try:
            keywords = [w for w, _ in lda_model.show_topic(topic_id, topn=10)]
        except Exception:
            keywords = []
        topic_keywords_map[f"topic_{topic_id}"] = ', '.join(keywords) if keywords else '无关键词'

        context_samples = []
        for tokens, summary in zip(group_df['tokens'], group_df['Summary']):
            if any(k in tokens for k in keywords):
                context_samples.append(summary)
            if len(context_samples) >= 3:
                break
        topic_text = ' '.join(keywords + context_samples)
        topic_texts.append(topic_text)

    # --------------------- 标签匹配 ---------------------
    label_names = list(raw_terms.keys())
    label_texts = [' '.join(set(raw_terms[label]['停用词'])) for label in label_names] if label_names else []

    topic_vecs = model.encode(topic_texts, convert_to_tensor=True)
    label_vecs = model.encode(label_texts, convert_to_tensor=True) if label_texts else []

    topic_labels = {}
    print(f"\n使用上下文增强+BERT向量获取Top-{top_n}标签：")

    for i in range(num_topics):
        if not len(label_vecs):
            topic_labels[i] = '未分类风险'
            continue
        sims = cosine_similarity(
            [topic_vecs[i].cpu().numpy()],
            [v.cpu().numpy() for v in label_vecs]
        )[0]
        top_indices = sims.argsort()[::-1][:top_n]
        matches = [(label_names[j], sims[j]) for j in top_indices if sims[j] >= min_score]
        topic_labels[i] = ', '.join(f"{lbl} ({score:.2f})" for lbl, score in matches) if matches else '未分类风险'

    group_df['risk_category'] = group_df['dominant_topic'].map(topic_labels)

    # --------------------- 输出结果 ---------------------
    output_dir = '01industry_outputs14'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"output_{number}_{industry_name}.csv")
    output_cols = ['Symbol', 'ShortName', 'DeclareDate', 'Summary',
                   'dominant_topic', 'risk_category'] + [f'topic_{i}' for i in range(num_topics)]
    group_df[output_cols].to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"已保存行业输出文件：{output_file}")

    topic_keywords_df = pd.DataFrame({
        'Topic': [f'topic_{i}' for i in range(num_topics)],
        'Top Words': [topic_keywords_map[f'topic_{i}'] for i in range(num_topics)],
        'Risk Label': [topic_labels[i] for i in range(num_topics)]
    })
    keyword_file = os.path.join(output_dir, f"topics_{number}_{industry_name}.csv")
    topic_keywords_df.to_csv(keyword_file, index=False, encoding='utf-8-sig')
    print(f"已保存主题关键词文件：{keyword_file}")
