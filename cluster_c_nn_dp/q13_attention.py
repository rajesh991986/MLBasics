"""
Q13 — Scaled Dot-Product Attention
Apple Sr. Applied ML Interview | CoderPad-safe (NumPy only)

APPROACH (say this in first 60 seconds):
  "I'll implement Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V.
   Key steps: transpose K using swapaxes on last two dims only (not K.T which
   reverses all axes), apply optional causal mask by setting future positions
   to -1e9 before softmax, then numerically stable softmax (subtract max),
   then weighted sum of V.
   Multi-head: split d_model into H heads via reshape+transpose, run same
   attention per head, combine back. I'll implement core first."

COMPLEXITY: Time O(S^2 * d_k) | Space O(S^2 + S*d)
"""

import numpy as np


# ─────────────────────────────────────────────
# CORE — write this first, always (5–8 min)
# ─────────────────────────────────────────────

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        Q : (..., seq_q, d_k)
        K : (..., seq_k, d_k)
        V : (..., seq_k, d_v)
        mask: bool array, True = BLOCKED (set to -1e9 before softmax)

    Returns:
        output  : (..., seq_q, d_v)
        weights : (..., seq_q, seq_k)  — rows sum to 1
    """
    d_k = Q.shape[-1]

    # swapaxes(-2,-1) only swaps seq<->d_k, keeps batch/heads intact
    # K.T would reverse ALL axes — wrong for batched tensors
    scores = Q @ np.swapaxes(K, -2, -1) / np.sqrt(d_k)   # (..., seq_q, seq_k)

    # mask=True means "blocked" — set to -1e9 so softmax gives ~0 weight
    # use -1e9 not -inf: avoids nan if an entire row is masked
    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # numerically stable softmax: subtract max before exp
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)         # (..., seq_q, seq_k)

    return weights @ V, weights                            # output, weights


# ─────────────────────────────────────────────
# MULTI-HEAD — only write if interviewer asks
# (explain verbally first, code on request)
# ─────────────────────────────────────────────

def multi_head_attention(x, W_q, W_k, W_v, W_o, num_heads, causal=False):
    """
    Multi-head self-attention.

    Args:
        x       : (B, S, D)  — input
        W_q/k/v : (D, D)     — projection weights
        W_o     : (D, D)     — output projection
        num_heads: int        — number of heads H
        causal  : bool        — apply causal (autoregressive) mask

    Returns:
        output  : (B, S, D)
        weights : (B, H, S, S)
    """
    B, S, D = x.shape
    d_k = D // num_heads

    def split(z):
        # (B, S, D) -> (B, H, S, d_k)
        return z.reshape(B, S, num_heads, d_k).transpose(0, 2, 1, 3)

    Q, K, V = split(x @ W_q), split(x @ W_k), split(x @ W_v)

    # causal mask: upper triangle = future positions = blocked
    # np.triu(..., k=1): True above main diagonal only
    mask = np.triu(np.ones((S, S), dtype=bool), k=1) if causal else None

    out, weights = scaled_dot_product_attention(Q, K, V, mask)

    # (B, H, S, d_k) -> (B, S, D)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    return out @ W_o, weights


# ─────────────────────────────────────────────
# TEST — minimal, one assert proves correctness
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    # ── Test 1: core attention, weights sum to 1 ──
    Q = K = V = np.random.randn(2, 5, 8)
    out, w = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (2, 5, 8),        f"bad output shape: {out.shape}"
    assert np.allclose(w.sum(axis=-1), 1), "weights don't sum to 1"
    print("Test 1 pass — core attention, weights sum to 1")

    # ── Test 2: causal mask blocks future positions ──
    S = 4
    causal_mask = np.triu(np.ones((S, S), dtype=bool), k=1)
    Q2 = K2 = V2 = np.random.randn(1, S, 8)
    _, wc = scaled_dot_product_attention(Q2, K2, V2, mask=causal_mask)
    # upper triangle must be ~0
    assert np.allclose(wc[0][causal_mask], 0, atol=1e-6), "future positions not masked"
    print("Test 2 pass — causal mask zeros out future positions")

    # ── Test 3: multi-head output shape ──
    B, S, D, H = 2, 5, 16, 4
    x = np.random.randn(B, S, D)
    scale = np.sqrt(2.0 / D)
    Wq, Wk, Wv, Wo = [np.random.randn(D, D) * scale for _ in range(4)]
    out_mh, w_mh = multi_head_attention(x, Wq, Wk, Wv, Wo, num_heads=H, causal=True)
    assert out_mh.shape == (B, S, D),       f"bad MHA output shape: {out_mh.shape}"
    assert w_mh.shape  == (B, H, S, S),     f"bad MHA weight shape: {w_mh.shape}"
    assert np.allclose(w_mh.sum(axis=-1), 1), "MHA weights don't sum to 1"
    print("Test 3 pass — multi-head shapes and weights correct")

    print("\nAll tests passed!")

    # ── Quick shape trace for interview explanation ──
    print("\n── Shape trace (B=2, H=4, S=5, D=16, d_k=4) ──")
    print(f"  Input x        : {x.shape}")
    print(f"  After project  : {(B, S, D)}")
    print(f"  After split    : {(B, H, S, D//H)}")
    print(f"  Scores Q@K^T   : {(B, H, S, S)}")
    print(f"  After @V       : {(B, H, S, D//H)}")
    print(f"  After combine  : {out_mh.shape}")