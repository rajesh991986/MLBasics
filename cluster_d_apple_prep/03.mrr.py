"""
MRR — Mean Reciprocal Rank

WHEN TO USE:
  Single-answer queries where one correct document exists.
  "Did we rank the right doc first?" — most Siri PQA queries are this type.
  Found at rank 1 → 1.0, rank 2 → 0.5, rank 5 → 0.2, never found → 0.

VS NDCG:
  MRR:   only looks at the FIRST relevant result. Ignores docs 2-5 even if relevant.
  NDCG:  considers ALL relevant docs at ALL ranks. Better for multi-doc queries.
  Use MRR for Siri simple queries (most common). Use NDCG as primary overall metric.

COMPLEXITY:
  Time:  O(N * K)  — N queries, scan K results each
  Space: O(|relevant|)  — set per query
"""


def mrr(all_retrieved, all_relevant):
    """
    Args:
        all_retrieved (list of lists): Retrieved doc IDs per query.
        all_relevant  (list of lists): Relevant doc IDs per query.
    Returns:
        float: Mean Reciprocal Rank across all queries.
    """
    if not all_retrieved:
        return 0.0

    total = 0.0
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        relevant_set = set(relevant)
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant_set:
                total += 1.0 / rank
                break           # only FIRST hit matters for MRR

    return total / len(all_retrieved)


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("MRR TESTS")
print("=" * 60)

# Test 1: All found at rank 1 → perfect MRR
result = mrr([["doc1", "doc2"], ["doc3", "doc4"]], [["doc1"], ["doc3"]])
assert result == 1.0, f"Expected 1.0, got {result}"
print(f"\n[Test 1] All at rank 1 → MRR = {result:.4f}")
print("✅ Test 1 PASSED")

# Test 2: Mixed ranks
all_ret = [["doc1", "doc3", "doc2"], ["doc5", "doc1", "doc2"], ["doc2", "doc1", "doc3"]]
all_rel = [["doc2"], ["doc2"], ["doc2"]]
# Query 1: doc2 at rank 3 → 1/3
# Query 2: doc2 at rank 3 → 1/3 (wait, doc2 is at index 2 = rank 3)
# Query 3: doc2 at rank 1 → 1/1
# MRR = (1/3 + 1/3 + 1/1) / 3 = (0.333 + 0.333 + 1.0) / 3 = 0.556
result = mrr(all_ret, all_rel)
expected = (1/3 + 1/3 + 1.0) / 3
assert abs(result - expected) < 1e-6
print(f"\n[Test 2] Mixed ranks (1/3, 1/3, 1/1) → MRR = {result:.4f}")
print("✅ Test 2 PASSED")

# Test 3: Never found → MRR = 0
result = mrr([["doc1", "doc2"]], [["doc99"]])
assert result == 0.0
print(f"\n[Test 3] Doc never retrieved → MRR = {result}")
print("✅ Test 3 PASSED")

# Test 4: Empty list
result = mrr([], [])
assert result == 0.0
print(f"\n[Test 4] Empty lists → MRR = {result}")
print("✅ Test 4 PASSED")

# Test 5: break after first hit (MRR ignores 2nd+ relevant docs)
result_single = mrr([["doc1", "doc2"]], [["doc1"]])  # one relevant, found at 1
result_two    = mrr([["doc1", "doc2"]], [["doc1", "doc2"]])  # two relevant, first at 1
assert result_single == result_two == 1.0, "MRR = 1.0 regardless of additional relevant docs"
print(f"\n[Test 5] 1 relevant vs 2 relevant, both at rank 1 → same MRR = {result_single}")
print("✅ Test 5 PASSED")

print("\n" + "=" * 60)
print("ALL 5 TESTS PASSED ✅")
print("=" * 60)
