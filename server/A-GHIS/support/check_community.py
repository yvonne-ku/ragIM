import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import networkx as nx

from server import settings

# ====================== 配置项（请根据你的实际路径修改） ======================
# 原始图的路径（和build_community.py中一致）
GRAPH_PKL_PATH = settings.basic_settings.SERVER_ROOT / "A-GHIS" / "workspace" / "graph.pkl"
# 社区数据的输出目录（和build_community.py中一致）
COMM_OUTPUT_DIR = settings.basic_settings.SERVER_ROOT / "A-GHIS" / "workspace"
# 你测试的其中一组参数（比如先选分辨率1.0、电导率0.2，替换成你实际跑过的参数）
TARGET_RESOLUTION = 1.0
TARGET_CONDUCTANCE = 0.2


# ====================== 核心分析函数 ======================
def load_data():
    """加载原始图和社区检测结果"""
    # 1. 加载原始图
    with open(GRAPH_PKL_PATH, 'rb') as f:
        G = pickle.load(f)
    print("=== 1. 原始图基础信息 ===")
    print(f"原始图节点数: {G.number_of_nodes()}")
    print(f"原始图边数: {G.number_of_edges()}")
    # 计算加权度统计
    weighted_degrees = dict(G.degree(weight='weight')).values()
    print(f"原始图加权度统计:")
    print(f"  平均加权度: {np.mean(list(weighted_degrees)):.2f}")
    print(f"  最小加权度: {np.min(list(weighted_degrees)):.2f}")
    print(f"  最大加权度: {np.max(list(weighted_degrees)):.2f}")
    # 统计孤立节点数（加权度为0）
    isolated_nodes = [n for n, d in G.degree(weight='weight') if d == 0]
    print(f"原始图孤立节点数（加权度=0）: {len(isolated_nodes)}")

    # 2. 加载社区检测结果（Pickle格式）
    comm_pkl_path = COMM_OUTPUT_DIR / "pkl" / f"R_{TARGET_RESOLUTION}_C_{TARGET_CONDUCTANCE}.pkl"
    if not comm_pkl_path.exists():
        raise FileNotFoundError(f"未找到社区数据文件: {comm_pkl_path}")
    with open(comm_pkl_path, 'rb') as f:
        comm_data = pickle.load(f)

    community_registry = comm_data['community_registry']
    frozen_communities = comm_data['frozen_communities']
    node_to_communities = comm_data['node_to_communities']
    community_hierarchy = comm_data['community_hierarchy']

    return G, community_registry, frozen_communities, node_to_communities, community_hierarchy


def analyze_community_distribution(community_registry):
    """分析社区注册表的分布特征"""
    print("\n=== 2. 社区注册表核心统计 ===")
    # 1. 总社区数 & 各层级社区数
    all_levels = [comm['level'] for comm in community_registry.values()]
    level_counter = Counter(all_levels)
    print(f"总社区数: {len(community_registry)}")
    print(f"各层级社区数分布: {dict(level_counter)}")
    print(f"最大层级: {max(all_levels) if all_levels else 0}")

    # 2. 冻结/非冻结社区数
    frozen_count = sum(1 for comm in community_registry.values() if comm['frozen'])
    unfrozen_count = len(community_registry) - frozen_count
    print(f"冻结社区数: {frozen_count}")
    print(f"非冻结社区数: {unfrozen_count}")

    # 3. 电导率分布
    conductances = [comm['conductance'] for comm in community_registry.values()]
    print(f"电导率统计:")
    print(f"  平均值: {np.mean(conductances):.4f}")
    print(f"  最小值: {np.min(conductances):.4f}")
    print(f"  最大值: {np.max(conductances):.4f}")
    print(f"  中位数: {np.median(conductances):.4f}")
    # 统计电导率≤阈值的社区数（验证冻结逻辑）
    threshold_hit = sum(1 for c in conductances if c <= TARGET_CONDUCTANCE)
    print(f"电导率≤{TARGET_CONDUCTANCE}的社区数: {threshold_hit}")

    # 4. 社区大小分布
    comm_sizes = [comm['size'] for comm in community_registry.values()]
    print(f"社区大小统计:")
    print(f"  平均大小: {np.mean(comm_sizes):.2f}")
    print(f"  最小大小: {np.min(comm_sizes)}")
    print(f"  最大大小: {np.max(comm_sizes)}")
    # 统计单节点社区数（可能是孤立节点导致）
    single_node_comm = sum(1 for s in comm_sizes if s == 1)
    print(f"单节点社区数: {single_node_comm}")


def verify_conductance_calculation(G, community_registry):
    """验证电导率计算的正确性（随机选5个社区复核）"""
    print("\n=== 3. 电导率计算复核（随机5个社区） ===")
    # 随机选5个不同层级的社区
    sample_comm_ids = list(community_registry.keys())[:5]
    for cid in sample_comm_ids:
        comm = community_registry[cid]
        nodes = comm['nodes']
        # 重新计算电导率
        total_volume = sum(dict(G.degree(weight='weight')).values())
        vol_community = sum(dict(G.degree(nodes, weight='weight')).values())
        subgraph = G.subgraph(nodes)
        internal_weight = sum(data['weight'] for _, _, data in subgraph.edges(data=True))
        cut_weight = max(0.0, vol_community - 2 * internal_weight)
        denominator = min(vol_community, total_volume - vol_community)

        if denominator == 0:
            recalc_phi = 0.0 if vol_community == total_volume else 1.0
        else:
            recalc_phi = cut_weight / denominator

        print(f"社区ID {cid} (层级{comm['level']}):")
        print(f"  原电导率: {comm['conductance']:.4f} | 重新计算: {recalc_phi:.4f}")
        print(
            f"  中间值: cut_weight={cut_weight:.2f}, vol_community={vol_community:.2f}, internal_weight={internal_weight:.2f}")
        print(f"  冻结状态: {comm['frozen']} (电导率阈值{TARGET_CONDUCTANCE})")


def analyze_hierarchy_iteration(community_registry):
    """分析层次迭代的社区合并趋势"""
    print("\n=== 4. 层次迭代社区数变化 ===")
    # 按层级统计活跃社区（非冻结）数
    level_to_comm = defaultdict(list)
    for comm in community_registry.values():
        level_to_comm[comm['level']].append(comm)

    for level in sorted(level_to_comm.keys()):
        comms = level_to_comm[level]
        frozen_in_level = sum(1 for c in comms if c['frozen'])
        unfrozen_in_level = len(comms) - frozen_in_level
        print(f"层级{level}: 总社区数={len(comms)} | 冻结={frozen_in_level} | 活跃（非冻结）={unfrozen_in_level}")


def main():
    try:
        # 加载数据
        G, community_registry, frozen_communities, node_to_communities, community_hierarchy = load_data()

        # 核心分析
        analyze_community_distribution(community_registry)
        verify_conductance_calculation(G, community_registry)
        analyze_hierarchy_iteration(community_registry)

        # 额外验证：冻结社区的电导率是否真的≤阈值
        print("\n=== 5. 冻结社区电导率验证 ===")
        frozen_conductances = [c['conductance'] for c in frozen_communities]
        invalid_frozen = [c for c in frozen_communities if c['conductance'] > TARGET_CONDUCTANCE]
        print(f"冻结社区电导率范围: {min(frozen_conductances):.4f} ~ {max(frozen_conductances):.4f}")
        print(f"电导率超过阈值的冻结社区数: {len(invalid_frozen)}")

    except Exception as e:
        print(f"分析过程出错: {e}")
        raise


if __name__ == "__main__":
    main()