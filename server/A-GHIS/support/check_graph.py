import pickle
import networkx as nx
import os

from server import settings

# ===================== 请改成你实际保存的 graph 文件路径 =====================
GRAPH_SAVE_PATH = settings.basic_settings.SERVER_ROOT / "A-GHIS" / "workspace" / "graph.pkl"  # 你保存图的路径，改成你自己的！
# ============================================================================

def check_graph_health():
    print("=" * 60)
    print("🔍 开始检查图文件健康度（点、边、连通性、权重）")
    print("=" * 60)

    # 1. 检查文件是否存在
    if not os.path.exists(GRAPH_SAVE_PATH):
        print(f"❌ 错误：图文件不存在！路径：{GRAPH_SAVE_PATH}")
        return

    # 2. 加载图
    try:
        with open(GRAPH_SAVE_PATH, 'rb') as f:
            G = pickle.load(f)
        print(f"✅ 成功加载图文件：{GRAPH_SAVE_PATH}")
    except Exception as e:
        print(f"❌ 加载图失败：{e}")
        return

    # 3. 基础统计
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(f"\n📊 基础统计：")
    print(f"   节点数量（实体数）：{num_nodes:,}")
    print(f"   边数量（关系数）：{num_edges:,}")

    if num_nodes == 0:
        print("❌ 错误：图中没有任何节点！")
        return
    if num_edges == 0:
        print("❌ 错误：图中没有任何边！全是孤立点 → 社区发现完全失效")
        return

    # 4. 检查孤立点（没有任何边的节点）
    isolated_nodes = list(nx.isolates(G))
    print(f"\n🚫 孤立点检查：")
    print(f"   孤立点数量：{len(isolated_nodes)}")
    if len(isolated_nodes) / num_nodes > 0.3:
        print(f"   ⚠️  警告：超过 30% 都是孤立点，社区发现会严重异常")

    # 5. 检查图连通性（最重要！）
    connected_components = list(nx.connected_components(G))
    num_components = len(connected_components)
    print(f"\� 连通分量检查（决定能不能聚合！）：")
    print(f"   连通分量数量：{num_components:,}")

    # 连通分量大小分布
    component_sizes = sorted([len(c) for c in connected_components], reverse=True)
    print(f"   最大连通分量大小：{component_sizes[0]}")
    print(f"   前10个连通分量大小：{component_sizes[:10]}")

    if num_components > 1000:
        print("   ❌ 严重问题：图被切成了上千个碎片 → 社区永远无法合并！")
    elif num_components > 300:
        print(f"   ❌ 致命问题：连通分量 = {num_components} → 这就是你冻结社区=370~380的元凶！")

    # 6. 边权重检查
    weights = [d.get('weight', 1) for u, v, d in G.edges(data=True)]
    if len(weights) > 0:
        print(f"\n⚖️ 边权重检查：")
        print(f"   平均权重：{sum(weights)/len(weights):.2f}")
        print(f"   最小权重：{min(weights)}")
        print(f"   最大权重：{max(weights)}")

    # 7. 总结
    print("\n" + "=" * 60)
    print("✅ 检查完成！把上面的结果发给我，我告诉你图是否正常")
    print("=" * 60)

if __name__ == "__main__":
    check_graph_health()