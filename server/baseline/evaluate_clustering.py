# evaluate_clustering_simple.py
import json
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from server import settings

def load_true_labels(label_path):
    """加载真实标签，返回 message_id -> topic_id 的字典"""
    true_labels_map = {}
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)["content"]
    for msg in data:
        msg_id = msg["msg_id"]
        topic_id = msg["topic_id"]
        true_labels_map[msg_id] = topic_id
    return true_labels_map

def load_pred_labels(chunk_path):
    """从分块文件中提取预测标签，返回 message_id -> chunk_id 的字典"""
    pred_labels_map = {}
    with open(chunk_path, 'r', encoding='utf-8') as f:
        data = json.load(f)["chunks"]
    for chunk in data:
        chunk_id = chunk["chunk_id"]
        msg_ids = chunk.get("msg_ids", [])
        for msg_id in msg_ids:
            pred_labels_map[msg_id] = chunk_id
    return pred_labels_map

def evaluate(true_labels, pred_labels, OUTPUT_DIR):
    """计算 NMI 和 ARI，要求两个字典的 key 集合一致"""
    # 1. 找出两个字典共同拥有的 msg_id (防止预测或真实数据有缺失)
    common_keys = set(true_labels.keys()) & set(pred_labels.keys())
    if not common_keys:
        print("错误：真实标签和预测标签没有共同的 msg_id，无法评估！")
        return 0.0, 0.0
    if len(common_keys) != len(true_labels) or len(common_keys) != len(pred_labels):
        print(
            f"警告：样本不完全匹配！真实: {len(true_labels)}, 预测: {len(pred_labels)}, 交集对齐后: {len(common_keys)}")

    # 2. 严格按照相同的 key 顺序提取 label 列表
    y_true = [true_labels[k] for k in common_keys]
    y_pred = [pred_labels[k] for k in common_keys]

    # 3. 计算指标 (sklearn 内部会自动处理绝对值不同、只看簇结构的问题)
    nmi_score = normalized_mutual_info_score(y_true, y_pred)
    ari_score = adjusted_rand_score(y_true, y_pred)
    res_dict = {
        "model": "baseline",
        "params": {
            "buffer_size": 1,
            "threshold_percentile": 90,
        },
        "nmi": nmi_score,
        "ari": ari_score
    }
    with open(OUTPUT_DIR, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    true_raw_file = settings.basic_settings.RAW_JSON_PATH
    chunk_file = settings.basic_settings.CHUNKS_DIR / "semantic_split_b_1_p_90.json"

    # 加载真实标签
    true_labels = load_true_labels(true_raw_file)
    print(f"已加载 {len(true_labels)} 条消息的真实话题标签")

    # 加载分块预测标签
    pred_labels = load_pred_labels(chunk_file)
    print(f"已加载 {len(pred_labels)} 条消息的预测分块标签")

    # 计算聚类评估指标
    OUTPUT_DIR = settings.basic_settings.RESULTS_DIR / "baseline_clustering.json"
    evaluate(true_labels, pred_labels, OUTPUT_DIR)
    print(f"聚类评估结果已保存到 {OUTPUT_DIR}")
