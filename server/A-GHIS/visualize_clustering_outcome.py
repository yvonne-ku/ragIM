import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def load_results(result_dir: Path):
    """加载所有评估结果文件，返回参数到指标的映射"""
    results = {}
    for file in result_dir.glob("A-GHIS_clustering_R*.json"):
        # 解析文件名中的分辨率 R 和电导率阈值 C
        # 文件名格式: A-GHIS_clustering_R1.0_C0.05.json
        try:
            parts = file.stem.split('_')
            r_part = [p for p in parts if p.startswith('R')][0]
            c_part = [p for p in parts if p.startswith('C')][0]
            resolution = float(r_part[1:])
            conductance = float(c_part[1:])
        except (IndexError, ValueError):
            print(f"跳过无法解析的文件: {file.name}")
            continue
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        nmi = data.get('nmi')
        ari = data.get('ari')
        if nmi is not None and ari is not None:
            results[(resolution, conductance)] = {'nmi': nmi, 'ari': ari}
    return results

def harmonic_mean(nmi, ari):
    if nmi + ari == 0:
        return 0
    return 2 * nmi * ari / (nmi + ari)

def main():
    # 指定聚类结果目录（根据你的实际路径修改）
    clustering_output_dir = Path(__file__).parent / "workspace" / "output"
    if not clustering_output_dir.exists():
        print(f"目录不存在: {clustering_output_dir}")
        return

    results = load_results(clustering_output_dir)
    if not results:
        print("未找到任何评估结果文件")
        return

    # 计算综合得分
    scored = []
    for (r, c), metrics in results.items():
        nmi = metrics['nmi']
        ari = metrics['ari']
        score = harmonic_mean(nmi, ari)
        scored.append((r, c, nmi, ari, score))

    # 排序找出最佳（综合得分最高）
    scored.sort(key=lambda x: x[4], reverse=True)
    best = scored[0]
    print("最佳参数组合（按调和平均得分）:")
    print(f"  分辨率 (resolution) = {best[0]}")
    print(f"  电导率阈值 (conductance) = {best[1]}")
    print(f"  NMI = {best[2]:.4f}")
    print(f"  ARI = {best[3]:.4f}")
    print(f"  调和平均得分 = {best[4]:.4f}")

    # 打印前5名
    print("\n前5名参数组合:")
    for i, (r, c, nmi, ari, score) in enumerate(scored[:5], 1):
        print(f"{i}. R={r}, C={c} -> NMI={nmi:.4f}, ARI={ari:.4f}, Score={score:.4f}")

    # 准备绘图数据
    resolutions = sorted(set(r for r, _ in results.keys()))
    conductances = sorted(set(c for _, c in results.keys()))
    nmi_matrix = np.full((len(resolutions), len(conductances)), np.nan)
    ari_matrix = np.full((len(resolutions), len(conductances)), np.nan)

    for i, r in enumerate(resolutions):
        for j, c in enumerate(conductances):
            if (r, c) in results:
                nmi_matrix[i, j] = results[(r, c)]['nmi']
                ari_matrix[i, j] = results[(r, c)]['ari']

    # 绘制热力图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # NMI 热力图
    im1 = ax1.imshow(nmi_matrix, cmap='viridis', aspect='auto', origin='lower',
                     extent=[conductances[0], conductances[-1], resolutions[0], resolutions[-1]])
    ax1.set_xlabel('Conductance Threshold')
    ax1.set_ylabel('Resolution')
    ax1.set_title('Normalized Mutual Information (NMI)')
    plt.colorbar(im1, ax=ax1)

    # ARI 热力图
    im2 = ax2.imshow(ari_matrix, cmap='plasma', aspect='auto', origin='lower',
                     extent=[conductances[0], conductances[-1], resolutions[0], resolutions[-1]])
    ax2.set_xlabel('Conductance Threshold')
    ax2.set_ylabel('Resolution')
    ax2.set_title('Adjusted Rand Index (ARI)')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(clustering_output_dir / 'hyperparam_heatmaps.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()