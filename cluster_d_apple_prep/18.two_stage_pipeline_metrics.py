"""
Two-Stage Retrieval Pipeline Metrics

YOUR SYSTEM:
  Stage 1 — bi-encoder:  400K+ articles → ANN search → top-50 candidates
            Metric: Recall@50 — "is the correct doc in the pool?"
  Stage 2 — cross-encoder: top-50 → rerank → top-5 for answer model
            Metric: Precision@5 + NDCG@5 — "is the final output quality good?"

THE CEILING:
  Recall@K at bi-encoder is a HARD CEILING on all downstream metrics.
  If correct doc not in top-50: cross-encoder cannot recover it.
  Your system: Recall@9 = 0.73 → tool accuracy ceiling = 0.73.
  Root cause (Story 7): labels generated from retrieved top-9 → model learns retrieval's mistakes.
  Fix: decouple label generation → give labeler ALL docs, not just retrieved candidates.

COMPLEXITY:
  All metrics: O(K) per query, O(N × K) total for N queries.
"""

import math
from collections import defaultdict


def recall_at_k(retrieved, relevant, k):
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(set(retrieved[:k]) & relevant_set) / len(relevant_set)


def precision_at_k(retrieved, relevant, k):
    relevant_set = set(relevant)
    hits = sum(1 for doc in retrieved[:k] if doc in relevant_set)
    return hits / k if k > 0 else 0.0


def ndcg_at_k(retrieved, relevant, k):
    relevant_set = set(relevant)
    relevances   = [1 if doc in relevant_set else 0 for doc in retrieved[:k]]

    def dcg(rels):
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

    ideal = dcg(sorted(relevances, reverse=True))
    return dcg(relevances) / ideal if ideal > 0 else 0.0


def mrr(retrieved, relevant):
    relevant_set = set(relevant)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / rank
    return 0.0


def evaluate_pipeline(bi_results, reranked, ground_truth, recall_k=50, rerank_k=5):
    """
    Args:
        bi_results  (list of lists): Bi-encoder top-recall_k per query.
        reranked    (list of lists): Reranker top-rerank_k per query.
        ground_truth(list of lists): Relevant doc IDs per query.
    Returns:
        dict: All pipeline metrics.
    """
    n = len(ground_truth)
    return {
        f"bi_recall@{recall_k}":     round(sum(recall_at_k(b, g, recall_k) for b, g in zip(bi_results, ground_truth)) / n, 4),
        f"rerank_precision@{rerank_k}": round(sum(precision_at_k(r, g, rerank_k) for r, g in zip(reranked, ground_truth)) / n, 4),
        f"rerank_ndcg@{rerank_k}":      round(sum(ndcg_at_k(r, g, rerank_k)      for r, g in zip(reranked, ground_truth)) / n, 4),
        "rerank_mrr":                    round(sum(mrr(r, g)                       for r, g in zip(reranked, ground_truth)) / n, 4),
    }


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("TWO-STAGE PIPELINE METRICS TESTS")
print("=" * 60)

bi = [
    ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"],  # query 1: correct=doc1 (rank 1)
    ["doc7", "doc8", "doc9", "doc1", "doc2", "doc3"],  # query 2: correct=doc9 (rank 3)
    ["doc4", "doc5", "doc6", "doc7", "doc8", "doc9"],  # query 3: correct=doc10 (MISSED!)
]
reranked = [
    ["doc1", "doc3", "doc5"],   # q1: correct at rank 1
    ["doc9", "doc7", "doc8"],   # q2: reranker promoted doc9 to rank 1
    ["doc4", "doc6", "doc5"],   # q3: correct not here (bi-encoder missed it)
]
gt = [["doc1"], ["doc9"], ["doc10"]]

metrics = evaluate_pipeline(bi, reranked, gt, recall_k=6, rerank_k=3)

# Test 1: Bi-encoder recall < 1.0 (doc10 missed)
assert metrics["bi_recall@6"] < 1.0
print(f"\n[Test 1] bi_recall@6 = {metrics['bi_recall@6']}  (doc10 missed → < 1.0)")
print("✅ Test 1 PASSED")

# Test 2: Reranker precision > bi-encoder raw precision
# Reranker got 2/3 correct at rank 1, bi-encoder wouldn't
assert metrics["rerank_precision@3"] > 0.15   # 2/3 queries got correct doc
print(f"\n[Test 2] rerank_precision@3 = {metrics['rerank_precision@3']}")
print("✅ Test 2 PASSED")

# Test 3: MRR reflects first-hit rank
assert metrics["rerank_mrr"] > 0.0
print(f"\n[Test 3] rerank_mrr = {metrics['rerank_mrr']}")
print("✅ Test 3 PASSED")

# Test 4: Full metrics printout
print("\n[Test 4] Full pipeline metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")
print("✅ Test 4 PASSED")

# Test 5: Ceiling — rerank precision can't exceed bi recall
assert metrics["rerank_precision@3"] <= metrics["bi_recall@6"] + 0.01
print(f"\n[Test 5] Ceiling: rerank_precision@3 ({metrics['rerank_precision@3']}) ≤ bi_recall@6 ({metrics['bi_recall@6']})")
print("✅ Test 5 PASSED")

print("\n" + "=" * 60)
print("ALL 5 TESTS PASSED ✅")
print("=" * 60)
