"""
Embedding 向量化验证脚本
=========================
运行方式：python test_embedder.py

测试内容：
    1. 将几条文本向量化
    2. 验证向量的维度和归一化结果
    3. 演示余弦相似度计算（语义相近 vs 语义无关）
"""

import sys
sys.path.insert(0, ".")

from src.rag.embedder import Embedder


def compute_dot_product(vec_a, vec_b):
    """
    计算两个向量的点积（内积）。

    点积 = a[0]*b[0] + a[1]*b[1] + ... + a[n]*b[n]

    由于我们的向量已经做了 L2 归一化，点积 = 余弦相似度。
    值域：-1（完全相反）到 1（完全相同），0 表示无关。
    """
    total = 0.0
    for a_val, b_val in zip(vec_a, vec_b):
        total = total + (a_val * b_val)
    return total


def compute_magnitude(vec):
    """
    计算向量的模长（L2 范数）。
    用于验证归一化结果：归一化后的模长应等于 1.0。
    """
    sum_sq = 0.0
    for v in vec:
        sum_sq = sum_sq + (v * v)
    return sum_sq ** 0.5  # 等同于 math.sqrt(sum_sq)


def main():
    print("=" * 60)
    print("Embedding 向量化验证")
    print("=" * 60)

    # 第一步：初始化 Embedder
    print("\n[1] 初始化 Embedder...")
    try:
        embedder = Embedder()
        print(f"    使用模型：{embedder._model}")
    except Exception as e:
        print(f"    ❌ 初始化失败：{e}")
        print("\n    常见原因：")
        print("    1. API Key 未在 .env 中配置")
        print("    2. 当前 API 供应商不支持 Embedding 端点")
        print("    3. 需要在 .env 中设置 EMBEDDING_BASE_URL 指向独立的 Embedding 服务")
        return

    # 第二步：准备测试文本
    # 故意使用了三组语义关系明确的文本对，方便观察相似度
    texts = [
        "苹果是一种很常见的水果，富含维生素",          # 文本 0：水果话题
        "香蕉也是水果，含有丰富的钾元素",              # 文本 1：水果话题（与 0 相近）
        "向量数据库用于存储和检索高维向量",            # 文本 2：数据库话题
        "Embedding 是将文本转换为数值向量的技术",      # 文本 3：数据库话题（与 2 相近）
    ]

    # 第三步：批量向量化
    print(f"\n[2] 正在对 {len(texts)} 条文本做 Embedding...")
    try:
        vectors = embedder.embed(texts)
        print(f"    完成！共 {len(vectors)} 个向量，每个 {len(vectors[0])} 维。")
    except Exception as e:
        print(f"    ❌ Embedding 失败：{type(e).__name__}: {e}")
        print("\n    故障排查建议：")
        print("    1. 检查 .env 中 EMBEDDING_MODEL 是否设置为你的供应商支持的模型")
        print("    2. 如果你的 API 供应商不提供 Embedding 服务，")
        print("       请在 .env 中设置 EMBEDDING_BASE_URL 指向独立的 Embedding API")
        print("    3. 可以尝试 EMBEDDING_MODEL=text-embedding-3-small（需 OpenAI Key）")
        return

    # 第四步：验证归一化
    print(f"\n[3] 验证 L2 归一化（每个向量的模长应 ≈ 1.0）：")
    for i, vec in enumerate(vectors):
        mag = compute_magnitude(vec)
        status = "✅" if abs(mag - 1.0) < 0.001 else "❌"
        print(f"    向量 #{i} 模长 = {mag:.6f}  {status}")

    # 第五步：计算语义相似度矩阵
    print(f"\n[4] 相似度矩阵（余弦相似度 = 点积，因为已归一化）：")
    print(f"    {'':>6}", end="")
    for i in range(len(vectors)):
        print(f"{'文本'+str(i):>10}", end="")
    print()

    for i in range(len(vectors)):
        print(f"    {'文本'+str(i):>4}", end="")
        for j in range(len(vectors)):
            sim = compute_dot_product(vectors[i], vectors[j])
            print(f"{sim:>10.4f}", end="")
        print()

    # 第六步：关键验证——语义相近的文本应该有更高的相似度
    print(f"\n[5] 语义判断验证：")
    sim_0_1 = compute_dot_product(vectors[0], vectors[1])  # 苹果 vs 香蕉（同属水果）
    sim_0_2 = compute_dot_product(vectors[0], vectors[2])  # 苹果 vs 向量数据库（无关）

    print(f"    '苹果是水果' vs '香蕉是水果'       → 相似度 = {sim_0_1:.4f}")
    print(f"    '苹果是水果' vs '向量数据库用于存储' → 相似度 = {sim_0_2:.4f}")

    if sim_0_1 > sim_0_2:
        print(f"    ✅ 语义相近的文本相似度更高（{sim_0_1:.4f} > {sim_0_2:.4f}），Embedding 工作正常！")
    else:
        print(f"    ⚠️  语义相近的文本相似度反而更低，可能需要检查 Embedding 模型质量。")

    print()
    print("=" * 60)
    print("✅ Embedding 验证完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()
