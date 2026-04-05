# -*- coding: gbk -*-
import os
# 设置环境变量强制离线
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import os
import re
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import jieba
import jieba.posseg as pseg


# 参数
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
N_CLUSTERS = 12            # 目标 risk type 数量
BALANCE_TOLERANCE = 0.20  # 允许每簇大小相对于目标平均值的偏差
MIN_CLUSTER_MIN = 3       # 最小允许簇大小
TOP_K_REP = 3             # 每簇用于命名的代表标签数
NAME_MAX_WORDS = 1        # 命名时最多保留的关键词数
# ----------------------------

model = SentenceTransformer(MODEL_NAME)

def extract_labels_from_folder(folder_path):
    """提取每个 CSV 第3列以后的列名（全局标签表头），并返回 industry 列表"""
    all_labels = set()
    industries = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".csv") or not fname.startswith("topic_label"):
            continue
        industries.append(os.path.splitext(fname)[0])
        df = pd.read_csv(os.path.join(folder_path, fname), encoding='gbk', header=0, engine='python')
        cols = list(df.columns)[2:]
        cleaned = []
        for c in cols:
            if pd.isna(c):
                continue
            s = str(c).strip()
            s = re.sub(r"[\(（]\s*[0-9.]+\s*[\)）]", "", s).strip()
            if s and not re.fullmatch(r"[0-9.]+", s):
                cleaned.append(s)
        all_labels.update(cleaned)
    return list(all_labels), industries

def embed_labels(labels):
    """批量生成 embeddings（返回 numpy array）"""
    emb = model.encode(labels, convert_to_tensor=True)
    return emb  # torch tensor OK for util.cos_sim; sklearn needs numpy

def initial_kmeans(labels, embeddings, n_clusters=N_CLUSTERS):
    """用 KMeans 得到初始簇及中心"""
    X = embeddings.cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    ids = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_  # numpy array
    clusters = defaultdict(list)
    for i, cid in enumerate(ids):
        clusters[int(cid)].append(labels[i])
    return clusters, centers, ids

def compute_centroid_embedding(labels_subset):
    """对 labels_subset 生成 embedding 的中心向量（numpy）"""
    if len(labels_subset) == 0:
        return None
    emb = model.encode(labels_subset, convert_to_tensor=True)
    centroid = emb.mean(dim=0).cpu().numpy()
    return centroid

def balance_clusters(labels, initial_clusters, centers, embeddings, desired_k=N_CLUSTERS,
                     tolerance=BALANCE_TOLERANCE, min_cluster_min=MIN_CLUSTER_MIN):
    """
    均衡化簇大小：
    - desired_avg = total_labels / desired_k
    - 允许每簇大小在 [desired_avg*(1-tol), desired_avg*(1+tol)]
    - 对小簇：把其成员按与其他簇 center 的相似度分配到最近的簇（并删除小簇）
    - 此过程尽量使簇大小均衡
    返回新的 clusters (dict risk_id -> label list) 和 centers (numpy array)
    """
    total = len(labels)
    desired_avg = total / desired_k
    low_thresh = max(min_cluster_min, int(np.floor(desired_avg*(1-tolerance))))
    high_thresh = int(np.ceil(desired_avg*(1+tolerance)))

    # 转为可操作结构： cluster_id -> label list
    clusters = {int(k): list(v) for k,v in initial_clusters.items()}

    # compute centers for current clusters
    cluster_centers = {}
    for k, lab_list in clusters.items():
        cluster_centers[k] = compute_centroid_embedding(lab_list)

    # 找到小簇列表（size < low_thresh）
    while True:
        sizes = {k: len(v) for k,v in clusters.items()}
        # 若所有簇都满足阈值或仅剩少于 desired_k 簇，停止
        small_clusters = [k for k,s in sizes.items() if s < low_thresh]
        if not small_clusters:
            break

        # 处理一个小簇：把它的 labels 分配到最相近（语义）的簇，并删除该簇
        k_small = small_clusters[0]
        lab_to_move = clusters.pop(k_small)
        center_small = cluster_centers.pop(k_small, None)

        # 若没有其他簇（极端情况），重新放回并退出
        if len(clusters) == 0:
            clusters[k_small] = lab_to_move
            cluster_centers[k_small] = center_small
            break

        # 计算每个 label 到现有簇中心的相似度，并分配到最佳簇
        for lab in lab_to_move:
            lab_emb = model.encode(lab, convert_to_tensor=True)
            best_k = None
            best_sim = -999
            for k_remain, c_emb in cluster_centers.items():
                if c_emb is None:
                    continue
                sim = util.cos_sim(lab_emb, c_emb).item()
                if sim > best_sim:
                    best_sim = sim
                    best_k = k_remain
            # assign
            clusters[best_k].append(lab)

        # 更新所有簇中心
        for k in list(clusters.keys()):
            cluster_centers[k] = compute_centroid_embedding(clusters[k])

    # 最后一步：如果簇数超过 desired_k（因为初始 k 可能等于 desired_k，但删除小簇后簇数下降），
    # 若簇数少于 desired_k，不强制拆分；若多于 desired_k, 也允许（但我们用 initial_kmeans 的 k）
    # 为简洁，返回当前 clusters 及 centers
    final_centers = {k: v for k,v in cluster_centers.items()}
    return clusters, final_centers

def extract_candidate_words(labels_list, top_n=30):
    """
    从 labels_list 中用 jieba 提取中文关键词候选（名词/短语），返回按频率排序的词表
    - 使用 pseg 提取词性，优先 n（名词）、nr、nz、vn等
    """
    counter = Counter()
    for lab in labels_list:
        # 分词并标注词性
        words = pseg.cut(lab)
        for w, flag in words:
            # 只保留中文词且长度>=2
            if re.fullmatch(r"[\u4e00-\u9fa5]+", w) and len(w) >= 2:
                # 过滤掉过泛的词（可扩展）
                if w in {"风险", "影响", "变化", "不及"}:
                    continue
                counter[w] += 1
    most = [w for w,_ in counter.most_common(top_n)]
    return most

def name_cluster_by_semantics(cluster_labels, cluster_centroid, top_k_rep=TOP_K_REP, max_words=NAME_MAX_WORDS):
    """
    给单个簇生成简洁语义名称：
    1) 取 cluster 的前 top_k_rep 个最代表标签（与 centroid 相似度最高）
    2) 从这些代表标签及整个簇中提取候选词
    3) 用 embedding 相似度对候选词与 centroid 做打分，选出 1-2 个词作为名称核心
    4) 组合为 'X风险' 或 'X与Y风险'
    """
    if not cluster_labels:
        return "其他风险"

    # 代表标签
    lab_embs = model.encode(cluster_labels, convert_to_tensor=True)
    centroid_emb = cluster_centroid if isinstance(cluster_centroid, np.ndarray) else cluster_centroid
    # 如果 centroid_emb 是 numpy array, 转成 tensor
    if isinstance(centroid_emb, np.ndarray):
        centroid_tensor = util.tensor_to_device(util.dot, model.device) if False else None  # no-op to keep compatibility
        centroid_tensor = None
    # we will compute cosine between centroid (numpy) and label emb by converting centroid->tensor
    import torch
    centroid_tensor = torch.from_numpy(cluster_centroid).to(lab_embs.device)

    sims = util.cos_sim(centroid_tensor, lab_embs).cpu().numpy().flatten()
    top_idx = list(np.argsort(-sims)[:min(len(cluster_labels), top_k_rep)])
    top_labels = [cluster_labels[i] for i in top_idx]

    # 候选词来自：top_labels + cluster_labels
    candidates = extract_candidate_words(top_labels, top_n=40)
    if len(candidates) < 1:
        candidates = extract_candidate_words(cluster_labels, top_n=40)

    # 若仍无候选词，直接用最代表的标签短化为名称（截断为2-3字）
    if not candidates:
        rep = top_labels[0]
        # 取首个连续中文短词作为名称
        kw = re.findall(r"[\u4e00-\u9fa5]{2,4}", rep)
        if kw:
            return f"{kw[0]}风险"
        else:
            return rep if len(rep) <= 6 else rep[:6] + "风险"

    # 将候选词用 embedding 与 centroid 比相似度，选择 top N
    cand_embs = model.encode(candidates, convert_to_tensor=True)
    sims2 = util.cos_sim(centroid_tensor, cand_embs).cpu().numpy().flatten()
    cand_scores = list(zip(candidates, sims2))
    cand_scores.sort(key=lambda x: -x[1])

    chosen = [w for w,_ in cand_scores[:max_words]]
    # 合并命名规则：若第一个词包含“市场/竞争/需求”等，优先单词+风险
    name_core = "与".join(chosen)
    # 如果过长，仅取第一个或前两个词
    chosen_short = chosen[:max_words]
    name_core = "与".join(chosen_short)
    return f"{name_core}风险"

# ==== 监督关键词集 ====
RISK_KEYWORDS = {
    "政策监管类": ["政策", "监管", "调控", "医保", "放松", "控费", "改革", "政府", "行业政策"],
    "成本与价格类": ["原材料", "饲料", "价格", "上涨", "成本", "费用", "毛利率", "运营成本"],
    "经营与盈利类": ["销售", "扩张", "扩店", "盈利", "利润", "整合", "门店", "项目注入"],
    "市场环境类": ["宏观经济", "地缘政治", "市场情绪", "需求复苏", "出口", "地产", "行业恢复"],
    "技术研发类": ["技术", "研发", "新产品", "创新", "装机", "推广", "失败"],
    "金融与资本类": ["IPO", "股价", "投资", "债务", "参股", "融资", "财费", "财富管理"],
    "自然与外部环境类": ["天气", "灾害", "环境", "海洋", "捕捞", "资源变化", "锡价", "国际客流"],
    "产能与进度类": ["项目", "进度", "投产", "投建", "装配式", "矿产资源", "基建", "爬坡", "产能", "节奏"],
    "风险通用词": ["不及预期", "超预期", "不确定", "拖累", "滞后", "下降", "下行", "波动"]
}

# ==== 计算监督中心 ====
def compute_supervised_centers(model):
    centers = {}
    for risk_name, keywords in RISK_KEYWORDS.items():
        emb = model.encode(keywords, convert_to_tensor=True)
        centers[risk_name] = emb.mean(dim=0, keepdim=True)  # 每类关键词的中心
    return centers

# ==== 替换 generate_risk_types_and_classify ====
def classify_with_supervision(labels, industries, model, threshold=0.35):
    """
    基于 RISK_KEYWORDS 的半监督风险分类。
    threshold: 相似度阈值（低于此值的标签归入 "其他风险"）
    分类完成后：若存在 "风险通用词" 或 "其他风险"，将其标签重新分配。
    """
    label_embs = model.encode(labels, convert_to_tensor=True)
    centers = compute_supervised_centers(model)

    classified = defaultdict(list)
    for i, label in enumerate(labels):
        sims = {}
        for cname, c_emb in centers.items():
            sims[cname] = util.cos_sim(label_embs[i], c_emb).item()
        best_name = max(sims, key=sims.get)
        best_sim = sims[best_name]

        if best_sim < threshold:
            classified["其他风险"].append(label)
        else:
            classified[best_name].append(label)

    # === 重新分配“风险通用词”和“其他风险”的标签 ===
    redistribute_keys = [k for k in classified.keys() if k in ("风险通用词", "其他风险")]
    if redistribute_keys:

        # 收集需要重新分配的标签
        to_redistribute = []
        for key in redistribute_keys:
            to_redistribute.extend(classified[key])
            del classified[key]

        # 仅保留有效类别中心
        valid_centers = {k: v for k, v in centers.items() if k not in redistribute_keys}

        # 将每个标签重新分配到最相似的风险类型
        for label in to_redistribute:
            emb = model.encode(label, convert_to_tensor=True)
            sims = {cname: util.cos_sim(emb, c_emb).item() for cname, c_emb in valid_centers.items()}
            best_cname = max(sims, key=sims.get)
            classified[best_cname].append(label)

        print(f"已重新分配 {len(to_redistribute)} 个标签。")

    # === 打印结果统计 ===
    total_labels = sum(len(v) for v in classified.values())
    print(f"分类完成，共 {total_labels} 个标签被归类：")
    for cname, labs in classified.items():
        print(f"  - {cname}: {len(labs)} 个标签")

    result = {
        "classified": dict(classified),
        "industries": industries
    }
    return result

def main():
        folder = "C:/Users/ASUS/Desktop/test/lda_for_paper/01topic_labels14"
        labels, industries = extract_labels_from_folder(folder)
        print(f"提取到 {len(labels)} 个标签")

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # 使用半监督分类而非聚类
        result = classify_with_supervision(labels, industries, model, threshold=0.35)

        with open("03risk_classification/risk_types_supervised14.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print("分类完成，结果保存为 risk_types_supervised14.json")

if __name__ == "__main__":
    main()



