import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Set, Dict, List, Tuple, Any, DefaultDict

import networkx as nx
import leidenalg as la
import igraph as ig

from server import settings


def compute_conductance(G: nx.Graph, community_nodes: Set) -> float:
    """
    计算给定社区的电导率，电导率是社区发现领域评估"高内聚低耦合"的核心指标：
    - 取值范围：[0, 1]
    - 0.0：完美社区（无任何对外连接，完全内聚）
    - 1.0：最差社区（无任何内部连接，完全离散）

    公式：φ = cut_weight / min(vol_community, total_volume - vol_community)
    其中：
    - cut_weight: 割边权重：社区与外部所有连接边的权重之和
    - vol_community: 社区体积：社区内所有节点的加权度之和
    - total_volume - vol_community: 补集体积

    Args:
        G: nx.Graph - 原始的无向加权全局图，边带有weight属性
        community_nodes: Set - 待评估社区包含的原始图节点ID集合

    Returns:
        float - 社区的电导率值，值越小表示社区内聚性越高
    """
    # 图的总体积
    total_volume = sum(dict(G.degree(weight='weight')).values())

    # 社区体积：所有节点的加权度之和（无向图中 = 2 * 内部边权重 + 割边权重）
    vol_community = sum(dict(G.degree(community_nodes, weight='weight')).values())

    # 社区内部边权重
    subgraph = G.subgraph(community_nodes)
    internal_weight = sum(data['weight'] for _, _, data in subgraph.edges(data=True))
    # 割边权重（社区对外的边权重）
    # 浮点精度防护，避免割边为负数
    cut_weight = max(0.0, vol_community - 2 * internal_weight)

    # 标准电导率分母
    denominator = min(vol_community, total_volume - vol_community)

    # ======================= 处理边界 case ==========================
    # 0.0 和 1.0 后续都直接 frozen = true 进行冻结
    # case 1: 空社区
    if not community_nodes:
        return 1.0
    # case 2: 全图社区
    if vol_community == total_volume:
        return 0.0
    # case 3: 零体积社区，所有节点都是孤立点
    if vol_community == 0:
        return 1.0

    phi = cut_weight / denominator
    return phi


def build_aggregation_graph(G: nx.Graph, comm_nodes_map: Dict[int, Set]) -> nx.Graph:
    """
    构建层次社区发现的聚合图（超级节点图）
    核心作用：将上一层聚类得到的社区抽象为"超级节点"，把原始图的边转换为超级节点之间的边
    用于下一轮Leiden算法的输入，实现自底向上的层次聚类

    Args:
        G: nx.Graph - 上一层的无向加权全局图，边带有weight属性
        comm_nodes_map: Dict[int, Set] - 上一层社区的映射关系
            键：社区ID（整数）
            值：该社区包含的原始图节点ID集合

    Returns:
        nx.Graph - 聚合后的超级节点图
            - 节点：上一层的社区ID
            - 边：两个社区之间所有跨社区原始边的权重之和
            - 边属性：weight（累加后的跨社区总权重）
    """
    agg_G = nx.Graph()

    # 辅助映射字典，上一层的节点ID -> 上一层的社区ID
    node_to_comm = {}
    for cid, nodes in comm_nodes_map.items():
        for node in nodes:
            node_to_comm[node] = cid

    # 添加上一层的社区节点作为聚合图的节点
    agg_G.add_nodes_from(comm_nodes_map.keys())

    # 将上一层的边，转化成这一层的边
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        cid_u = node_to_comm.get(u)
        cid_v = node_to_comm.get(v)
        if cid_u is not None and cid_v is not None and cid_u != cid_v:
            if agg_G.has_edge(cid_u, cid_v):
                agg_G[cid_u][cid_v]['weight'] += w
            else:
                agg_G.add_edge(cid_u, cid_v, weight=w)
    return agg_G


def run_leiden_on_agg_graph(agg_G: nx.Graph, resolution_parameter: float) -> List[List[int]]:
    """
    在层次社区发现的聚合图（超级节点图）上运行Leiden算法，返回新的社区划分结果
    这里使用RBConfigurationVertexPartition（配置模型顶点分区），是加权图社区发现的标准选择

    Args:
        agg_G: nx.Graph - 这一层的聚合图（超级节点图）
            节点：上一层的社区ID（整数）
            边：两个社区之间的跨社区总权重
        resolution_parameter: float - 分辨率参数，控制社区粒度
            - 越大：生成的社区数量越多、粒度越细
            - 越小：生成的社区数量越少、粒度越粗
            - 推荐范围：0.1 ~ 2.0，默认1.0对应标准模块度

    Returns:
        List[List[int]] - 新的社区划分结果
            外层列表：所有新社区
            内层列表：每个新社区包含的超级节点ID（即上一层的社区ID）
            例如：[[0, 2, 5], [1, 3], [4]] 表示将上一层的0、2、5号社区合并为一个新社区
    """
    # 边界case：空聚合图，直接返回空列表
    if agg_G.number_of_nodes() == 0:
        return []

    # 边界case：聚合图只有1个节点，直接返回单社区（无需运行算法）
    if agg_G.number_of_nodes() == 1:
        return [[list(agg_G.nodes())[0]]]

    # 1. 将NetworkX图转换为igraph图
    ig_graph = ig.Graph.from_networkx(agg_G)

    # 2. 保留原始节点ID的name属性
    ig_graph.vs['name'] = list(agg_G.nodes())

    # 3. 运行Leiden算法进行社区划分
    partition = la.find_partition(
        ig_graph,
        la.RBConfigurationVertexPartition,
        resolution_parameter=resolution_parameter,
        # 固定随机种子，保证每次运行结果完全一致（可复现性）
        seed=42,
        # 迭代次数：-1表示运行至算法完全收敛，避免默认2次迭代导致的结果不稳定
        n_iterations=-1
    )

    # 4. 解析算法输出，将结果转换为我们需要的格式：{新社区ID: [包含的超级节点ID]}
    membership = partition.membership
    community_dict = {}
    for vid, comm_id in enumerate(membership):
        original_node_id = ig_graph.vs[vid]["name"]
        community_dict.setdefault(comm_id, []).append(original_node_id)

    return list(community_dict.values())


def init_level0_communities(
    G: nx.Graph,
    resolution_parameter: float,
    next_comm_id: int
) -> Tuple[Dict[int, Dict[str, Any]], int, Dict[int, Set[Any]], Dict[int, int]]:
    """初始化Level 0的初始社区，拆分原函数中重复逻辑，提高复用性"""
    community_registry: Dict[int, Dict[str, Any]] = {}
    comm_nodes_map: Dict[int, Set[Any]] = {}
    comm_level_map: Dict[int, int] = {}

    # 运行Leiden算法获取初始分区
    ig_graph = ig.Graph.from_networkx(G)
    ig_graph.vs['name'] = list(G.nodes())
    partition0 = la.find_partition(
        ig_graph,
        la.RBConfigurationVertexPartition,
        resolution_parameter=resolution_parameter
    )

    # 解析分区结果，构建社区注册表
    for comm_membership in partition0:
        nodes = {ig_graph.vs[vid]["name"] for vid in comm_membership}
        phi = compute_conductance(G, nodes)
        community_registry[next_comm_id] = {
            'id': next_comm_id,
            'level': 0,
            'nodes': nodes,
            'conductance': round(phi, 4),  # 保留4位小数，减少存储冗余
            'frozen': False,
            'parent_ids': [],
            'child_ids': [],
            'size': len(nodes)  # 新增：社区节点数，便于检索过滤
        }
        comm_nodes_map[next_comm_id] = nodes
        comm_level_map[next_comm_id] = 0
        next_comm_id += 1

    return community_registry, next_comm_id, comm_nodes_map, comm_level_map


def detect_all_communities_hierarchical(
    G: nx.Graph,
    resolution_parameter: float,
    conductance_threshold: float,
    max_iterations: int
) -> tuple[
    dict[int, dict[str, Any]], list[dict[str, Any]], defaultdict[Any, list[dict[str, Any]]], dict[int, list[int]]]:

    print(f"=== 开始层次社区检测 ===")
    print(f"参数：分辨率={resolution_parameter} | 电导率阈值={conductance_threshold} | 最大迭代={max_iterations}")

    # 初始化变量
    next_comm_id = 0
    community_registry, next_comm_id, comm_nodes_map, comm_level_map = init_level0_communities(
        G, resolution_parameter, next_comm_id
    )
    active_comm_ids = set(community_registry.keys())
    print(f"Level 0 初始社区数: {len(active_comm_ids)}")

    # 迭代构建层次社区
    iteration = 0
    while iteration < max_iterations and len(active_comm_ids) > 1:
        print(f"\n--- 迭代 {iteration + 1} ---")
        print(f"当前活跃社区数: {len(active_comm_ids)}")

        # 构建聚合图
        active_nodes_map = {cid: comm_nodes_map[cid] for cid in active_comm_ids}
        agg_G = build_aggregation_graph(G, active_nodes_map)
        print(f"聚合图节点数: {agg_G.number_of_nodes()} | 边数: {agg_G.number_of_edges()}")

        # 边界1：聚合图节点数≤1，停止迭代
        if agg_G.number_of_nodes() <= 1:
            print("聚合图节点数≤1，终止迭代")
            break
        # 边界2：聚合图边数≤1，停止迭代
        if agg_G.number_of_edges() <= 1:
            print("聚合图边数≤1，终止迭代")
            for cid in active_comm_ids:
                community_registry[cid]['frozen'] = True
            break

        # 运行Leiden算法获取新分组
        new_groups = run_leiden_on_agg_graph(agg_G, resolution_parameter)
        print(f"新社区分组数: {len(new_groups)}")

        # 边界3：Leiden聚类已经收敛，停止迭代
        if len(new_groups) == len(active_comm_ids):
            print("聚类已完全收敛，无社区可合并，剩余社区全部冻结并终止迭代")
            for cid in active_comm_ids:
                community_registry[cid]['frozen'] = True
            break

        # 处理新分组，构建更高层级社区
        next_active_comm_ids = set()
        for group in new_groups:
            # 合并子社区的节点
            merged_nodes = set()
            child_comm_ids = []
            for old_cid in group:
                merged_nodes.update(comm_nodes_map[old_cid])
                child_comm_ids.append(old_cid)

            # 计算新社区的元信息
            max_child_level = max(comm_level_map[cid] for cid in child_comm_ids)
            new_level = max_child_level + 1
            phi = compute_conductance(G, merged_nodes)
            frozen = phi <= conductance_threshold
            new_cid = next_comm_id
            print(f"新社区 {new_cid} 合并 {len(child_comm_ids)} 个子社区，电导率 {phi:.4f}，是否冻结 {frozen}")


            # 注册新社区
            community_registry[new_cid] = {
                'id': new_cid,
                'level': new_level,
                'nodes': merged_nodes,
                'conductance': round(phi, 4),
                'frozen': frozen,
                'parent_ids': [],
                'child_ids': child_comm_ids,
                'size': len(merged_nodes)
            }
            # 更新子社区的父节点
            for child_cid in child_comm_ids:
                community_registry[child_cid]['parent_ids'].append(new_cid)

            # 更新映射表
            comm_nodes_map[new_cid] = merged_nodes
            comm_level_map[new_cid] = new_level
            next_comm_id += 1

            # 非冻结社区加入下一轮活跃列表
            if not frozen:
                next_active_comm_ids.add(new_cid)

        # 更新迭代状态
        active_comm_ids = next_active_comm_ids
        iteration += 1
        print(f"迭代 {iteration} 结束，下一轮活跃社区数: {len(active_comm_ids)}")


    # ========== 构建适配检索的输出 ==========
    # 1. 冻结社区列表（过滤+排序，便于检索）
    frozen_communities = sorted(
        [comm for comm in community_registry.values() if comm['frozen']],
        key=lambda x: (x['level'], -x['size'])  # 按层级升序、社区大小降序排序
    )

    # 2. 节点-社区映射（结构化，支持按层级/电导率检索）
    node_to_communities: DefaultDict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for cid, comm in community_registry.items():
        for node in comm['nodes']:
            node_to_communities[node].append({
                'community_id': cid,
                'level': comm['level'],
                'conductance': comm['conductance'],
                'frozen': comm['frozen'],
                'community_size': comm['size']
            })
    # 对每个节点的社区列表按层级排序（便于检索时按层级筛选）
    for node, comm_list in node_to_communities.items():
        comm_list.sort(key=lambda x: x['level'])

    # 3. 社区层级映射（快速溯源祖先社区，便于检索时向上聚合）
    community_hierarchy: Dict[int, List[int]] = {}
    for cid, comm in community_registry.items():
        ancestors = []
        current_parents = comm['parent_ids']
        while current_parents:
            parent_id = current_parents[0]  # 取第一个父节点（层次聚类通常单父）
            ancestors.append(parent_id)
            current_parents = community_registry[parent_id]['parent_ids']
        community_hierarchy[cid] = ancestors

    # 最终日志输出
    print(f"\n=== 社区检测完成 ===")
    print(f"总社区数（所有层级）: {len(community_registry)}")
    print(f"冻结社区数: {len(frozen_communities)}")
    print(f"最大社区层级: {max(comm['level'] for comm in community_registry.values()) if community_registry else 0}")
    return community_registry, frozen_communities, node_to_communities, community_hierarchy


def save_community_data_for_retrieval(
    output_dir: Path,
    resolution: float,
    conductance: float,
    community_registry: Dict[int, Dict[str, Any]],
    frozen_communities: List[Dict[str, Any]],
    node_to_communities: Dict[Any, List[Dict[str, Any]]],
    community_hierarchy: Dict[int, List[int]]
) -> None:
    """
    将社区数据持久化，适配后续检索阶段的存储格式
    """
    # 创建目录
    pkl_path = output_dir / "pkl" / f"R_{resolution}_C_{conductance}.pkl"
    json_path = output_dir / "json" / f"R_{resolution}_C_{conductance}.json"

    # 1. 处理数据（Set转List，便于JSON序列化）
    def convert_to_json_compatible(data: Any) -> Any:
        if isinstance(data, set):
            return list(data)
        elif isinstance(data, dict):
            return {k: convert_to_json_compatible(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_to_json_compatible(v) for v in data]
        else:
            return data

    # 2. 持久化Pickle（高效，保留原生类型）
    pkl_data = {
        'metadata': {
            'resolution': resolution,
            'conductance_threshold': conductance,
            'total_communities': len(community_registry),
            'frozen_communities_count': len(frozen_communities),
            'max_level': max(comm['level'] for comm in community_registry.values()) if community_registry else 0
        },
        'community_registry': community_registry,
        'frozen_communities': frozen_communities,
        'node_to_communities': node_to_communities,
        'community_hierarchy': community_hierarchy,
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 3. 持久化JSON（跨语言，便于检索）
    json_data = {
        'metadata': pkl_data['metadata'],
        'community_registry': convert_to_json_compatible(community_registry),
        'frozen_communities': convert_to_json_compatible(frozen_communities),
        'node_to_communities': convert_to_json_compatible(node_to_communities),
        'community_hierarchy': community_hierarchy,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"数据已保存至：")
    print(f"  Pickle: {pkl_path}")
    print(f"  JSON: {json_path}")


def calculate_clustering_metrics(
    G: nx.Graph,
    community_registry: Dict[int, Dict[str, Any]],
    node_to_communities: DefaultDict[Any, List[Dict[str, Any]]],
    community_hierarchy: Dict[int, List[int]],
    resolution: float,
    conductance_threshold: float,
    chunk_json_path: Path,
    true_labels_path: Path,
    output_dir: Path
) -> None:
    """
    评估 frozen 社区的聚类质量。

    步骤：
    1. 加载分块数据，建立 chunk_id -> msg_ids 映射。
    2. 加载真实话题标签。
    3. 对于每个 frozen 社区，收集其覆盖的所有 msg_id。
    4. 为每个 msg_id 分配一个预测社区 ID（多数投票，平局时选最小 ID）。
    5. 计算 NMI 和 ARI。
    6. 保存评估结果到 JSON 文件。
    """
    import json
    from collections import Counter
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    # ---------- 1. 加载分块数据 ----------
    with open(chunk_json_path, 'r', encoding='utf-8') as f:
        chunk_list = json.load(f)["chunks"]
    chunk_to_msgs = {}
    for ch in chunk_list:
        chunk_id = ch["chunk_id"]
        msg_ids = ch.get("msg_ids", [])
        chunk_to_msgs[chunk_id] = msg_ids

    # ---------- 2. 加载真实标签 ----------
    def load_true_labels(label_path):
        true_labels_map = {}
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)["content"]
        for msg in data:
            msg_id = msg["msg_id"]
            topic_id = msg["topic_id"]
            true_labels_map[msg_id] = topic_id
        return true_labels_map

    true_labels = load_true_labels(true_labels_path)
    print(f"已加载真实话题标签：{len(true_labels)} 条消息")

    # ---------- 3. 收集 frozen 社区覆盖的 msg_id ----------
    frozen_communities = [comm for comm in community_registry.values() if comm['frozen']]
    comm_to_msgs: Dict[int, List[str]] = {}  # frozen_comm_id -> list of msg_id

    for comm in frozen_communities:
        comm_id = comm['id']
        nodes = comm['nodes']   # 实体节点集合
        msg_set = set()
        for node in nodes:
            # 获取实体节点对应的 chunk_id 集合
            source_ids = G.nodes[node].get('source_ids', set())
            for chunk_id in source_ids:
                msg_ids = chunk_to_msgs.get(chunk_id, [])
                msg_set.update(msg_ids)
        comm_to_msgs[comm_id] = list(msg_set)

    # ---------- 4. 为每个 msg_id 分配唯一的预测社区 ID ----------
    # 统计每个 msg 出现在哪些 frozen 社区中（次数）
    msg_to_comm_counter: Dict[str, Counter] = defaultdict(Counter)
    for comm_id, msg_list in comm_to_msgs.items():
        for msg in msg_list:
            msg_to_comm_counter[msg][comm_id] += 1

    pred_labels = {}
    for msg, counter in msg_to_comm_counter.items():
        # 找出出现次数最多的社区 ID（平局时选最小的 ID）
        max_count = max(counter.values())
        most_common = [cid for cid, cnt in counter.items() if cnt == max_count]
        pred_labels[msg] = min(most_common)   # 选择最小的 ID 保证确定性
        # 注：一个消息可能属于多个 frozen 社区，但根据预期 frozen 社区互斥且覆盖实体，这里采用多数投票。

    # ---------- 5. 对齐真实标签与预测标签 ----------
    common_msgs = set(true_labels.keys()) & set(pred_labels.keys())
    if not common_msgs:
        print("错误：真实标签与预测标签没有共同的消息，无法评估")
        return

    y_true = [true_labels[msg] for msg in common_msgs]
    y_pred = [pred_labels[msg] for msg in common_msgs]

    # ---------- 6. 计算指标 ----------
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    # ---------- 7. 保存结果 ----------
    result = {
        "model": "A-GHIS",
        "params": {
            "resolution": resolution,
            "conductance_threshold": conductance_threshold,
        },
        "nmi": round(nmi, 4),
        "ari": round(ari, 4),
        "covered_messages": len(common_msgs),
    }

    out_file = output_dir / f"A-GHIS_clustering_R{resolution}_C{conductance_threshold}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"聚类评估结果已保存至 {out_file}")
    print(f"NMI = {nmi:.4f}, ARI = {ari:.4f}")


if __name__ == "__main__":
    true_labels_path = settings.basic_settings.RAW_JSON_PATH
    chunk_path = settings.basic_settings.CHUNKS_DIR / "semantic_split_b_1_p_90.json"
    output_dir = settings.basic_settings.SERVER_ROOT / "A-GHIS" / "workspace"
    clustering_output_dir = output_dir / "output"
    clustering_output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "graph.pkl", 'rb') as f:
        G = pickle.load(f)

    # 遍历参数组合
    resolution_list = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]
    conductance_list = [0.05,0.10,0.15,0.20,0.25,0.30]
    for r in resolution_list:
        for c in conductance_list:
            # 运行社区检测函数
            community_registry, frozen_communities, node_to_communities, community_hierarchy = detect_all_communities_hierarchical(
                G,
                resolution_parameter=r,
                conductance_threshold=c,
                max_iterations=7
            )
            # 持久化数据
            save_community_data_for_retrieval(
                output_dir=output_dir,
                resolution=r,
                conductance=c,
                community_registry=community_registry,
                frozen_communities=frozen_communities,
                node_to_communities=node_to_communities,
                community_hierarchy=community_hierarchy
            )
            calculate_clustering_metrics(
                G=G,
                community_registry=community_registry,
                node_to_communities=node_to_communities,
                community_hierarchy=community_hierarchy,
                resolution=r,
                conductance_threshold=c,
                chunk_json_path=chunk_path,
                true_labels_path=true_labels_path,
                output_dir=clustering_output_dir
            )
