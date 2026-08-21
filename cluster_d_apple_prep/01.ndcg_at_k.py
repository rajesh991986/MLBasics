"""
NDCG@K — Normalized Discounted Cumulative Gain

WHEN TO USE:
  Primary retrieval metric when rank position matters — e.g., Siri PQA where
  the answer model has a fixed 4K context window. A relevant doc at rank 8 may
  get truncated; NDCG discounts it accordingly.

VS OTHER METRICS:
  MRR:      only cares about rank of FIRST relevant doc. Good for single-answer Siri queries.
  Recall@K: did we find all relevant docs? Ignores rank. Good for bi-encoder ceiling check.
  NDCG@K:   both: finds relevant docs AND rewards early rank. Best unified primary metric.

COMPLEXITY:
  Time:  O(k log k)  — sorting for ideal DCG
  Space: O(k)

PRODUCTION (Apple PQA):
  Set K = floor(context_budget / avg_doc_tokens), not a round number.
  3B on-device model: 4K context, email ≈ 400 tokens → K=10 max.
  NDCG@10 is the right metric, not NDCG@100.
"""

import math


def dcg(relevances):
    """Discounted Cumulative Gain. log2(i+2) because rank is 1-indexed so i=0 → log2(2)=1."""
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))


def ndcg_at_k(relevances, k):
    """
    Normalized DCG at K.
    Args:
        relevances (list): Relevance scores of retrieved docs in rank order.
                           Graded (0-3) or binary (0/1) both work.
        k (int): Cutoff position.
    Returns:
        float: Score in [0, 1]. 1.0 = ideal ranking.
    """
    relevances = relevances[:k]
    actual = dcg(relevances)
    ideal  = dcg(sorted(relevances, reverse=True))
    return actual / ideal if ideal > 0 else 0.0


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("NDCG@K TESTS")
print("=" * 60)

# Test 1: Ideal ranking
result = ndcg_at_k([3, 2, 1, 0], 4)
assert abs(result - 1.0) < 1e-6, f"Expected 1.0, got {result}"
print(f"\n[Test 1] Ideal ranking → {result:.4f}")
print("✅ Test 1 PASSED")

# Test 2: Reversed (worst) ranking
result = ndcg_at_k([0, 1, 2, 3], 4)
assert result < 1.0
print(f"\n[Test 2] Worst ranking → {result:.4f} (< 1.0)")
print("✅ Test 2 PASSED")

# Test 3: All irrelevant
result = ndcg_at_k([0, 0, 0], 3)
assert result == 0.0
print(f"\n[Test 3] All irrelevant → {result}")
print("✅ Test 3 PASSED")

# Test 4: Binary relevance — correct doc at rank 1
result = ndcg_at_k([1, 0, 0, 0, 0], 5)
assert result == 1.0
print(f"\n[Test 4] Binary, correct at rank 1 → {result:.4f}")
print("✅ Test 4 PASSED")

# Test 5: Rank penalty — correct doc buried at rank 3
result = ndcg_at_k([0, 0, 1, 0, 0], 5)
# actual DCG = 1/log2(4) = 0.5, ideal DCG = 1/log2(2) = 1.0 → NDCG = 0.5
assert abs(result - 0.5) < 1e-6, f"Expected 0.5, got {result}"
print(f"\n[Test 5] Correct doc at rank 3 → {result:.4f} (rank penalty applied)")
print("✅ Test 5 PASSED")

# Test 6: k > list length — slice handles gracefully
result = ndcg_at_k([1, 0], 10)
assert result == 1.0
print(f"\n[Test 6] k > list length → {result:.4f} (no crash)")
print("✅ Test 6 PASSED")

print("\n" + "=" * 60)
print("ALL 6 TESTS PASSED ✅")
print("=" * 60)
