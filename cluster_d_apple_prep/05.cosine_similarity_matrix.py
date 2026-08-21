"""
Cosine Similarity Matrix

WHEN TO USE:
  Core operation in bi-encoder retrieval and InfoNCE loss.
  Given query embeddings A and document embeddings B, returns all pairwise similarities.

KEY TRICK:
  L2-normalize both matrices first → dot product = cosine similarity.
  Faster than computing cosine per-pair and avoids explicit division by magnitudes.

COMPLEXITY:
  Time:  O(n × m × d)  — one matrix multiply (highly optimized in numpy/BLAS)
  Space: O(n × m)      — output similarity matrix

PRODUCTION:
  At query time: embed query (n=1) → dot with all passage embeddings (m=400K).
  At training time (InfoNCE): all pairs in a batch (n=m=batch_size).
  + 1e-8 prevents division by zero for zero vectors.
  keepdims=True ensures shape (n,1) for broadcasting — without it, broadcast fails.
"""

import numpy as np


def cosine_similarity_matrix(A, B):
    """
    Args:
        A (np.ndarray): Shape (n, dim)
        B (np.ndarray): Shape (m, dim)
    Returns:
        np.ndarray: Shape (n, m) where [i,j] = cosine_sim(A[i], B[j])
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_norm @ B_norm.T


def cosine_similarity(a, b):
    """Cosine similarity between two 1D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("COSINE SIMILARITY MATRIX TESTS")
print("=" * 60)

# Test 1: Same direction → 1.0
A = np.array([[1.0, 0.0]])
B = np.array([[2.0, 0.0]])   # same direction, different magnitude
result = cosine_similarity_matrix(A, B)
assert abs(result[0, 0] - 1.0) < 1e-6
print(f"\n[Test 1] Same direction → sim = {result[0,0]:.4f}")
print("✅ Test 1 PASSED")

# Test 2: Perpendicular → 0.0
A = np.array([[1.0, 0.0]])
B = np.array([[0.0, 1.0]])
result = cosine_similarity_matrix(A, B)
assert abs(result[0, 0]) < 1e-6
print(f"\n[Test 2] Perpendicular → sim = {result[0,0]:.4f}")
print("✅ Test 2 PASSED")

# Test 3: 45-degree angle → ~0.707
A = np.array([[1.0, 0.0]])
B = np.array([[1.0, 1.0]])
result = cosine_similarity_matrix(A, B)
assert abs(result[0, 0] - 1/np.sqrt(2)) < 1e-6
print(f"\n[Test 3] 45-degree angle → sim = {result[0,0]:.4f}  (expect {1/np.sqrt(2):.4f})")
print("✅ Test 3 PASSED")

# Test 4: Self-similarity diagonal = 1.0
batch = np.random.randn(4, 64)
self_sim = cosine_similarity_matrix(batch, batch)
assert np.allclose(np.diag(self_sim), 1.0, atol=1e-6)
print(f"\n[Test 4] Self-similarity diagonal = {np.diag(self_sim).round(4)}")
print("✅ Test 4 PASSED")

# Test 5: Zero vector → no nan/inf
zero   = np.zeros((1, 4))
normal = np.random.randn(1, 4)
result = cosine_similarity_matrix(zero, normal)
assert not np.isnan(result).any() and not np.isinf(result).any()
print(f"\n[Test 5] Zero vector → sim = {result[0,0]:.4f}  (no nan/inf due to +1e-8)")
print("✅ Test 5 PASSED")

# Test 6: Matrix output shape
A = np.random.randn(3, 32)
B = np.random.randn(5, 32)
result = cosine_similarity_matrix(A, B)
assert result.shape == (3, 5)
print(f"\n[Test 6] A(3,32) × B(5,32) → output shape = {result.shape}")
print("✅ Test 6 PASSED")

print("\n" + "=" * 60)
print("ALL 6 TESTS PASSED ✅")
print("=" * 60)
