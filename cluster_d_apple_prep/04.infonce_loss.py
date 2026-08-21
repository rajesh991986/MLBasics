"""
InfoNCE Loss — Contrastive Learning for Embedding Models

WHEN TO USE:
  Training bi-encoder embedding models (ModernBERT-GTE, E5, Qwen3-Embedding).
  Teaches the model: query embedding ≈ matching passage embedding.
  Your system: used to train the embedding model backing 400K+ article retrieval.

HOW IT WORKS:
  In-batch negatives: for a batch of N query-passage pairs, each query uses
  all other N-1 passages as negatives for free.
  Positive pair = diagonal of similarity matrix. Loss = multiclass cross-entropy.

TEMPERATURE τ = 0.02 (GTE/E5 standard):
  Low τ → logits scaled by 1/0.02 = 50× → very sharp softmax distribution.
  Even 0.01 cosine difference becomes 0.5 logit difference → strong gradient signal.
  High τ (0.5) → soft distribution → model doesn't learn fine distinctions.

COMPLEXITY:
  Time:  O(B² × d) where B=batch, d=dim  — matmul dominates
  Space: O(B²)  — similarity matrix

PRODUCTION:
  CoderPad: use numpy (no torch). Real systems use torch + F.cross_entropy.
  Add query:/passage: prefixes BEFORE computing embeddings — they shift vectors
  into different subspaces that are calibrated to be close for matching pairs.
"""

import numpy as np
import math


def l2_normalize(X):
    """L2 normalize each row. After this, dot product = cosine similarity."""
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def infonce_loss(q_emb, p_emb, temperature=0.02):
    """
    Args:
        q_emb (np.ndarray): Query embeddings  (batch, dim)
        p_emb (np.ndarray): Passage embeddings (batch, dim). q_emb[i] ↔ p_emb[i].
        temperature (float): Softmax sharpness. Default 0.02 (GTE standard).
    Returns:
        float: Mean InfoNCE loss across the batch.
    """
    q = l2_normalize(q_emb)
    p = l2_normalize(p_emb)

    logits = (q @ p.T) / temperature              # (batch, batch)

    # Log-sum-exp trick: subtract row max before exp to prevent overflow
    row_max = logits.max(axis=1, keepdims=True)
    log_sum = np.log(np.exp(logits - row_max).sum(axis=1)) + row_max.squeeze()

    # Loss = -log P(correct passage | query). Correct = diagonal.
    loss = -(np.diag(logits) - log_sum)
    return float(loss.mean())


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("INFONCE LOSS TESTS")
print("=" * 60)
np.random.seed(42)

# Test 1: Perfect match (q == p) → loss near 0
q = np.random.randn(4, 64)
loss = infonce_loss(q, q.copy())
assert loss < 0.01, f"Expected ~0, got {loss:.4f}"
print(f"\n[Test 1] Perfect match (q == p) → loss = {loss:.6f}")
print("✅ Test 1 PASSED")

# Test 2: Random passages → high loss
p_rand = np.random.randn(4, 64)
loss = infonce_loss(q, p_rand)
assert loss > 0.5, f"Expected high loss, got {loss:.4f}"
print(f"\n[Test 2] Random passages → loss = {loss:.4f} (high)")
print("✅ Test 2 PASSED")

# Test 3: Low τ gives higher loss on hard examples (stronger signal)
loss_low  = infonce_loss(q, p_rand, temperature=0.01)
loss_high = infonce_loss(q, p_rand, temperature=0.5)
assert loss_low > loss_high, "Low τ should penalise wrong answers more"
print(f"\n[Test 3] τ=0.01: {loss_low:.4f}  τ=0.5: {loss_high:.4f}  (low τ = sharper)")
print("✅ Test 3 PASSED")

# Test 4: Batch size 1 → no negatives → loss = 0
q1 = np.random.randn(1, 64)
loss = infonce_loss(q1, q1.copy())
assert abs(loss) < 1e-6, f"Expected 0 (no negatives), got {loss}"
print(f"\n[Test 4] Batch=1, no negatives → loss = {loss:.6f}")
print("✅ Test 4 PASSED")

# Test 5: Numerical stability with large values
q_big = np.ones((4, 64)) * 1000
loss = infonce_loss(q_big, q_big.copy())
assert not math.isnan(loss) and not math.isinf(loss)
print(f"\n[Test 5] Large values (overflow test) → loss = {loss:.6f} (no nan/inf)")
print("✅ Test 5 PASSED")

print("\n" + "=" * 60)
print("ALL 5 TESTS PASSED ✅")
print("=" * 60)
