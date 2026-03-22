import numpy as np

np.random.seed(42)

# ─────────────────────────────────────────
# Dimensions (tiny so output is readable)
# ─────────────────────────────────────────
d_model = 4   # embedding size  → each token is a [4] vector
d_k     = 3   # projection size → Q, K, V are [3] vectors

# ─────────────────────────────────────────
# Learned weight matrices (fixed after training)
# ─────────────────────────────────────────
W_Q = np.random.randn(d_model, d_k)   # [4 × 3]
W_K = np.random.randn(d_model, d_k)   # [4 × 3]
W_V = np.random.randn(d_model, d_k)   # [4 × 3]

# ─────────────────────────────────────────
# Fake token embeddings  (normally from an embedding table)
# ─────────────────────────────────────────
vocab = {
    "The":      np.array([1.0, 0.0, 0.0, 0.0]),
    "doctor":   np.array([0.8, 0.6, 0.0, 0.2]),
    "treated":  np.array([0.0, 0.2, 0.9, 0.1]),
    "the":      np.array([1.0, 0.0, 0.1, 0.0]),
    "patient":  np.array([0.7, 0.5, 0.1, 0.3]),
    "because":  np.array([0.0, 0.0, 0.4, 0.8]),
    "she":      np.array([0.6, 0.8, 0.0, 0.1]),
    "was":      np.array([0.1, 0.1, 0.8, 0.2]),
    "kind":     np.array([0.5, 0.7, 0.2, 0.4]),
}

# ─────────────────────────────────────────
# KV Cache — this is the thing being explained
# Two lists that grow as tokens are processed
# ─────────────────────────────────────────
kv_cache = {
    "keys":   [],   # one K vector per token seen so far
    "values": [],   # one V vector per token seen so far
    "tokens": [],   # just for printing — not part of real cache
}

def compute_kv(token_name):
    """Compute K and V for one token and add to cache."""
    x = vocab[token_name]           # embedding [d_model]
    k = x @ W_K                     # project → [d_k]
    v = x @ W_V                     # project → [d_k]
    kv_cache["keys"].append(k)
    kv_cache["values"].append(v)
    kv_cache["tokens"].append(token_name)
    return k, v

def attention(query_token_name):
    """
    Given the NEW token being generated:
      1. Compute its Q (never cached)
      2. Load K, V from cache (never recomputed)
      3. Score Q against every cached K
      4. Weighted sum of V → output
    """
    x = vocab[query_token_name]
    q = x @ W_Q                                    # [d_k] — always fresh

    K_matrix = np.stack(kv_cache["keys"])          # [n_tokens × d_k]
    V_matrix = np.stack(kv_cache["values"])        # [n_tokens × d_k]

    # Score: how much does Q match each K?
    scores = q @ K_matrix.T                        # [n_tokens]
    scores = scores / np.sqrt(d_k)                 # scale by √d_k

    # Softmax → attention weights (sum to 1)
    scores -= scores.max()                         # numerical stability
    weights = np.exp(scores)
    weights /= weights.sum()                       # [n_tokens]

    # Weighted sum of V
    output = weights @ V_matrix                    # [d_k]
    return q, weights, output

# ════════════════════════════════════════
# STEP 1 — Process the prompt
# Compute K, V for every prompt token → fill the cache
# ════════════════════════════════════════
prompt = ["The", "doctor", "treated", "the", "patient", "because", "she"]

print("=" * 60)
print("STEP 1 — Processing prompt → filling KV cache")
print("=" * 60)

for token in prompt:
    k, v = compute_kv(token)
    print(f"  [{token:10s}]  K={np.round(k,2)}  V={np.round(v,2)}  → cached ✓")

print(f"\nCache size after prompt: {len(kv_cache['keys'])} tokens")
print(f"Cached tokens: {kv_cache['tokens']}")

# ════════════════════════════════════════
# STEP 2 — Generate "was"
# Q is computed fresh for "was"
# K, V are READ from cache (not recomputed)
# Then K, V of "was" itself is added to cache
# ════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — Generating token: 'was'")
print("=" * 60)

q, weights, output = attention("was")

print(f"\n  Q('was') = {np.round(q, 2)}   ← computed fresh, NOT cached")
print(f"\n  Attention weights over cached tokens:")
for token, w in zip(kv_cache["tokens"], weights):
    bar = "█" * int(w * 30)
    print(f"    {token:10s}  {w:.3f}  {bar}")

print(f"\n  Output for 'was' = {np.round(output, 2)}  (weighted blend of cached V vectors)")

# Now add "was" K, V to cache
compute_kv("was")
print(f"\n  K, V of 'was' → added to cache")
print(f"  Cache size now: {len(kv_cache['keys'])} tokens")

# ════════════════════════════════════════
# STEP 3 — Generate "kind"
# Same pattern — Q fresh, K/V from cache
# Note: "The", "doctor" etc. K,V are NOT recomputed
# ════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — Generating token: 'kind'")
print("=" * 60)

q, weights, output = attention("kind")

print(f"\n  Q('kind') = {np.round(q, 2)}   ← computed fresh, NOT cached")
print(f"\n  Attention weights over cached tokens:")
for token, w in zip(kv_cache["tokens"], weights):
    bar = "█" * int(w * 30)
    print(f"    {token:10s}  {w:.3f}  {bar}")

print(f"\n  Output for 'kind' = {np.round(output, 2)}")

# "The", "doctor" etc. were NEVER recomputed above
compute_kv("kind")

# ════════════════════════════════════════
# KEY INSIGHT — print the summary
# ════════════════════════════════════════
print("\n" + "=" * 60)
print("KEY INSIGHT — what was recomputed vs reused")
print("=" * 60)
print("""
  Token       Turn 2 ("was")     Turn 3 ("kind")
  ─────────────────────────────────────────────
  The         K,V from cache ✓   K,V from cache ✓
  doctor      K,V from cache ✓   K,V from cache ✓
  treated     K,V from cache ✓   K,V from cache ✓
  the         K,V from cache ✓   K,V from cache ✓
  patient     K,V from cache ✓   K,V from cache ✓
  because     K,V from cache ✓   K,V from cache ✓
  she         K,V from cache ✓   K,V from cache ✓
  was         K,V computed once  K,V from cache ✓
  kind        —                  Q only (generating)

  Q is NEVER cached — always computed fresh for the new token.
  K, V are computed ONCE and reused forever after.
""")