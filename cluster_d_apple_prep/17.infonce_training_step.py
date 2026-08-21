"""
InfoNCE Training Step — Embedding Model Training

WHAT THIS SHOWS:
  A complete contrastive training step in numpy (no torch for CoderPad).
  In production: replace numpy with torch + F.normalize + AdamW.

TRAINING RECIPE (your production system):
  Base model: ModernBERT or Qwen3-Embedding
  Loss: InfoNCE, τ=0.02, batch=256+
  Prefixes: "query: " on queries, "passage: " on passages (every example)
  Hard negatives: BM25 mined + dense mined (ANCE after round 2)
  Result: recall@9 = 0.73 on 400K+ article index

COMPLEXITY:
  Forward:  O(B × d) encode + O(B² × d) similarity matrix
  Backward: O(B² × d) gradient computation
  B=batch size, d=embedding dim
"""

import numpy as np
import math


# ── Core functions ────────────────────────────────────────────────────────────

def l2_normalize(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def infonce_loss_and_grad(q_emb, p_emb, temperature=0.02):
    """
    Returns:
        loss (float): Mean InfoNCE loss.
        grad_q (np.ndarray): Gradient w.r.t. query embeddings.
    """
    q = l2_normalize(q_emb)
    p = l2_normalize(p_emb)
    n = q.shape[0]

    logits  = (q @ p.T) / temperature           # (n, n)
    row_max = logits.max(axis=1, keepdims=True)
    exp_l   = np.exp(logits - row_max)
    softmax = exp_l / exp_l.sum(axis=1, keepdims=True)

    # Loss
    log_sum = np.log(exp_l.sum(axis=1)) + row_max.squeeze()
    loss    = -(np.diag(logits) - log_sum).mean()

    # Gradient: (softmax - one_hot) / (n × τ)
    one_hot    = np.eye(n)
    grad_q     = ((softmax - one_hot) / (n * temperature)) @ p

    return loss, grad_q


def training_step(q_emb, p_emb, lr=0.01, temperature=0.02):
    """One SGD step. Returns updated q_emb and loss."""
    loss, grad_q = infonce_loss_and_grad(q_emb, p_emb, temperature)
    q_emb_updated = q_emb - lr * grad_q
    return q_emb_updated, loss


# ============================================================================
# TESTS
# ============================================================================

print("=" * 60)
print("INFONCE TRAINING STEP TESTS")
print("=" * 60)
np.random.seed(42)

# Test 1: Loss decreases over training steps
q = np.random.randn(8, 32)
p = np.random.randn(8, 32)

losses = []
for _ in range(10):
    q, loss = training_step(q, p, lr=0.1)
    losses.append(loss)

assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
print(f"\n[Test 1] Loss after 10 steps: {losses[0]:.4f} → {losses[-1]:.4f} (decreasing)")
print("✅ Test 1 PASSED")

# Test 2: Perfect match → near-zero loss
q_perf = np.random.randn(4, 64)
loss, _ = infonce_loss_and_grad(q_perf, q_perf.copy())
assert float(loss) < 0.01
print(f"\n[Test 2] Perfect match loss = {loss:.6f} (~0)")
print("✅ Test 2 PASSED")

# Test 3: Gradient shape matches input
q = np.random.randn(4, 32)
p = np.random.randn(4, 32)
_, grad = infonce_loss_and_grad(q, p)
assert grad.shape == q.shape
print(f"\n[Test 3] Gradient shape = {grad.shape} (matches q_emb)")
print("✅ Test 3 PASSED")

print("\n" + "=" * 60)
print("ALL 3 TESTS PASSED ✅")
print("=" * 60)

print("\nIn PyTorch (real system):")
print("  q = encoder(query_ids)             # forward pass")
print("  p = encoder(passage_ids)           # forward pass")
print("  q = F.normalize(q, dim=-1)")
print("  p = F.normalize(p, dim=-1)")
print("  logits = (q @ p.T) / 0.02")
print("  labels = torch.arange(q.size(0))")
print("  loss = F.cross_entropy(logits, labels)")
print("  loss.backward()                    # auto-grad")
print("  optimizer.step()                   # AdamW update")
