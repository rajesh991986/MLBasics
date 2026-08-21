"""
Recall@K and Precision@K — Retrieval Metrics

WHEN TO USE:
  Recall@K:    bi-encoder stage — "is the correct doc in our candidate pool?"
               This is a HARD CEILING. If correct doc not in top-K, nothing downstream helps.
               Your system: Recall@9 = 0.73 → ceiling on overall tool accuracy.

  Precision@K: reranker output — "of the K docs passed to the answer model, how many are useful?"
               Only meaningful when queries have MULTIPLE relevant docs (your CS QA system).
               Apple PQA single-answer queries → Precision@K misleading (max = 1/K = 0.20 for K=5).

COMPLEXITY:
  Time:  O(K)    — iterate top-K
  Space: O(|relevant|)  — set for O(1) lookup

PRODUCTION (Apple PQA):
  Two-stage pipeline: measure Recall@50 at bi-encoder, Precision@5 + NDCG@5 at reranker.
  French threshold 0.52 vs English 0.60 — per-locale calibration needed because
  verbose French articles produce naturally lower cosine similarities.
"""

import numpy as np


def recall_at_k(retrieved, relevant, k):
    """
    Args:
        retrieved (list): Doc IDs in retrieval rank order.
        relevant  (list): All truly relevant doc IDs for this query.
        k (int): Cutoff position.
    Returns:
        float: Fraction of relevant docs found in top-K. Range [0, 1].
    """
    if not relevant:
        return 0.0
    top_k        = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(top_k & relevant_set) / len(relevant_set)


def precision_at_k(retrieved, relevant, k):
    """
    Args:
        retrieved (list): Doc IDs in retrieval rank order.
        relevant  (list): All truly relevant doc IDs.
        k (int): Cutoff position.
    Returns:
        float: Fraction of top-K docs that are relevant. Range [0, 1].
    """
    if k == 0:
        return 0.0
    top_k        = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc in top_k if doc in relevant_set)
    return hits / k


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("RECALL@K AND PRECISION@K TESTS")
print("=" * 60)

retrieved = ["doc1", "doc3", "doc2", "doc5", "doc7"]
relevant  = ["doc1", "doc2", "doc4"]

# Test 1: Recall@5 — 2 of 3 relevant docs found
result = recall_at_k(retrieved, relevant, 5)
assert abs(result - 2/3) < 1e-6, f"Expected {2/3:.4f}, got {result}"
print(f"\n[Test 1] Recall@5 = {result:.4f}  (doc1✓ doc2✓ doc4✗)")
print("✅ Test 1 PASSED")

# Test 2: Precision@5 — 2 hits in top 5
result = precision_at_k(retrieved, relevant, 5)
assert abs(result - 2/5) < 1e-6, f"Expected 0.4, got {result}"
print(f"\n[Test 2] Precision@5 = {result:.4f}  (2 relevant in top-5)")
print("✅ Test 2 PASSED")

# Test 3: Recall improves with larger K
r3 = recall_at_k(retrieved, relevant, 2)
r5 = recall_at_k(retrieved, relevant, 5)
assert r3 <= r5, "Recall can only go UP as K increases"
print(f"\n[Test 3] Recall@2={r3:.4f} ≤ Recall@5={r5:.4f} (monotonically non-decreasing)")
print("✅ Test 3 PASSED")

# Test 4: Precision can go DOWN with larger K (more noise)
p2 = precision_at_k(retrieved, relevant, 2)
p5 = precision_at_k(retrieved, relevant, 5)
print(f"\n[Test 4] Precision@2={p2:.4f} vs Precision@5={p5:.4f} (may decrease)")
print("✅ Test 4 PASSED")

# Test 5: Empty relevant set
result = recall_at_k(retrieved, [], 5)
assert result == 0.0
print(f"\n[Test 5] Empty relevant set → {result}")
print("✅ Test 5 PASSED")

# Test 6: Perfect recall
result = recall_at_k(["doc1", "doc2", "doc4"], ["doc1", "doc2", "doc4"], 3)
assert result == 1.0
print(f"\n[Test 6] Perfect recall → {result:.4f}")
print("✅ Test 6 PASSED")

# Test 7: Apple single-answer — Precision@5 misleading
print(f"\n[Test 7] Apple PQA single-answer: Precision@5 max = {1/5:.2f}")
print("         Only 1 correct doc exists → always ≤ 0.20 even if system is perfect")
print("         → Use Recall@K for Apple PQA, not Precision@K")
print("✅ Test 7 PASSED")

print("\n" + "=" * 60)
print("ALL 7 TESTS PASSED ✅")
print("=" * 60)
