"""
Q28 — From-Scratch Functions [SOMETIMES ASKED AS WARM-UP]
Target time: 15 min | Requires: numpy, math, collections

SAY THIS FIRST (30 sec):
  "I'll state the formula and edge cases before writing code,
   then verify the result against numpy/sklearn."

REPORTED ASKS: rmse, mae, sample variance, weighted sampler,
               precision/recall/F1, softmax, sigmoid, cosine similarity.
"""

import numpy as np
import math
from collections import Counter


# ══════════════════════════════════════════════════════════════
# SECTION 1: REGRESSION METRICS
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1: Regression Metrics")
print("=" * 60)

def rmse(y_true, y_pred):
    """sqrt( mean( (y-ŷ)² ) ) — penalises large errors (squares them). O(n)"""
    n  = len(y_true)
    ss = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    return math.sqrt(ss / n)

def mae(y_true, y_pred):
    """mean( |y-ŷ| ) — robust to outliers, same units as target. O(n)"""
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)

def mape(y_true, y_pred, eps=1e-8):
    """mean( |y-ŷ|/|y| ) × 100 — interpretable % error. Guard eps for y=0. O(n)"""
    return 100 * sum(
        abs(t - p) / (abs(t) + eps) for t, p in zip(y_true, y_pred)
    ) / len(y_true)

def r_squared(y_true, y_pred):
    """1 - SS_res/SS_tot.  1=perfect | 0=predict mean | <0=worse than mean"""
    mean_y = sum(y_true) / len(y_true)
    ss_tot = sum((t - mean_y) ** 2 for t in y_true)            # total variance
    ss_res = sum((t - p)      ** 2 for t, p in zip(y_true, y_pred))  # unexplained
    return 1 - ss_res / (ss_tot + 1e-9)

y_true = [3.0, 4.5, 2.1, 5.8, 3.3]
y_pred = [2.8, 4.7, 2.5, 5.2, 3.0]

print(f"  RMSE: {rmse(y_true,y_pred):.4f}   MAE: {mae(y_true,y_pred):.4f}")
print(f"  MAPE: {mape(y_true,y_pred):.2f}%  R²:  {r_squared(y_true,y_pred):.4f}")
# verify
print(f"  NumPy RMSE: {np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2)):.4f} ✓")


# ══════════════════════════════════════════════════════════════
# SECTION 2: STATISTICS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: Statistics")
print("=" * 60)

def sample_variance(data):
    """Σ(x-μ)²/(n-1) — divide by n-1 not n (Bessel's correction).
    WHY n-1: estimating population variance from a sample.
    Dividing by n underestimates; n-1 corrects the bias (lost 1 degree of
    freedom because we estimated the mean from the same data)."""
    n    = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / (n - 1)

def sample_std(data):
    return math.sqrt(sample_variance(data))

def median(data):
    """Sort then pick middle. O(n log n)."""
    s = sorted(data)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2     # even length: average two middles

def mode(data):
    """Most frequent value. O(n) via Counter."""
    return Counter(data).most_common(1)[0][0]

def percentile(data, p):
    """Linear interpolation. O(n log n)."""
    s   = sorted(data)
    n   = len(s)
    idx = (n - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)  # interpolate between neighbours

data = [4, 7, 13, 2, 8, 1, 9, 4, 7, 4, 3, 12]
print(f"  Data: {sorted(data)}")
print(f"  Variance: {sample_variance(data):.4f}  (numpy ddof=1: {np.var(data,ddof=1):.4f})")
print(f"  Std:      {sample_std(data):.4f}  (numpy: {np.std(data,ddof=1):.4f})")
print(f"  Median:   {median(data):.1f}    P75: {percentile(data,75):.2f}  Mode: {mode(data)}")


# ══════════════════════════════════════════════════════════════
# SECTION 3: CLASSIFICATION METRICS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3: Precision / Recall / F1")
print("=" * 60)

def precision_recall_f1(y_true, y_pred, pos_label=1):
    """
    Precision = TP/(TP+FP)  — of predictions that are +, how many correct?
    Recall    = TP/(TP+FN)  — of actual +, how many did we catch?
    F1        = 2PR/(P+R)   — harmonic mean: only high when BOTH are high

    WHY harmonic not arithmetic mean?
      P=1.0, R=0.01 → arithmetic=0.505 (misleadingly high), harmonic=0.02 (honest)
    """
    tp = sum(1 for t,p in zip(y_true,y_pred) if t==pos_label and p==pos_label)
    fp = sum(1 for t,p in zip(y_true,y_pred) if t!=pos_label and p==pos_label)
    fn = sum(1 for t,p in zip(y_true,y_pred) if t==pos_label and p!=pos_label)

    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return round(prec,4), round(rec,4), round(f1,4)

y_tc = [1,0,1,1,0,1,0,0,1,0]
y_pc = [1,0,1,0,0,1,1,0,1,0]
p,r,f = precision_recall_f1(y_tc, y_pc)
print(f"  Precision: {p}  Recall: {r}  F1: {f}")

from sklearn.metrics import precision_score, recall_score, f1_score
print(f"  sklearn:   {precision_score(y_tc,y_pc):.4f} / {recall_score(y_tc,y_pc):.4f} / {f1_score(y_tc,y_pc):.4f} ✓")


# ══════════════════════════════════════════════════════════════
# SECTION 4: NEURAL NET BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 4: NN Building Blocks")
print("=" * 60)

def sigmoid(x):
    """1/(1+e^-x).  Any real → (0,1).  Used in binary output layers."""
    return 1 / (1 + math.exp(-x))

def sigmoid_vec(x):
    """Numerically stable vector sigmoid (avoids exp overflow for large negatives)."""
    x = np.array(x, dtype=float)
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),          # stable for positive x
                    np.exp(x) / (1 + np.exp(x)))   # stable for negative x

def softmax(logits):
    """e^xᵢ / Σe^xⱼ → probability distribution (sums to 1). Multi-class output.
    Stability trick: subtract max before exp → largest value becomes e^0=1,
    all others shrink. Result is identical (constant cancels in numerator/denom)."""
    logits  = np.array(logits, dtype=float)
    shifted = logits - logits.max()   # shift so max=0, prevents overflow
    exps    = np.exp(shifted)
    return exps / exps.sum()

def relu(x):
    """max(0, x). Dead neurons for x<0, identity for x>0. No vanishing gradient."""
    return max(0.0, x)

print(f"  sigmoid(0)={sigmoid(0):.4f}  sigmoid(2)={sigmoid(2):.4f}  sigmoid(-10)={sigmoid(-10):.6f}")
print(f"  softmax([1,2,3])={softmax([1,2,3]).round(4)}  sum={softmax([1,2,3]).sum():.1f}")
print(f"  softmax([1000,1001,1002])={softmax([1000,1001,1002]).round(4)}  (stable ✓)")
print(f"  relu(-3)={relu(-3)}  relu(5)={relu(5)}")


# ══════════════════════════════════════════════════════════════
# SECTION 5: WEIGHTED RANDOM SAMPLER
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 5: Weighted Random Sampler")
print("=" * 60)
print("USE CASE: pick items with probability ∝ popularity/weight.")
print("PRODUCTION: np.random.choice(items, p=normalised_weights)")
print("FROM SCRATCH: normalise → cumsum → uniform random → binary search\n")

def weighted_sample(items, weights, k, replace=False):
    """Sample k items proportional to weights. O(k log n)."""
    weights = [w / sum(weights) for w in weights]  # normalise to sum=1
    cum, running = [], 0.0
    for w in weights:
        running += w
        cum.append(running)                         # cumulative distribution

    selected = []
    for _ in range(k):
        u = np.random.random()                      # U ~ Uniform(0,1)
        # binary search: find first bin where cum[idx] >= u
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < u: lo = mid + 1
            else:            hi = mid
        selected.append(items[lo])
        if not replace:                             # remove selected item
            items   = [x for i,x in enumerate(items)   if i != lo]
            weights = [w for i,w in enumerate(weights) if i != lo]
            s = sum(weights)
            cum, running = [], 0.0
            for w in weights:
                running += w/s if s>0 else 0
                cum.append(running)
    return selected

hotels = ["Budget Inn","City Hotel","Grand Palace","Boutique Stay","Airport Lodge"]
scores = [2, 5, 10, 7, 3]   # Grand Palace most popular → sampled most

np.random.seed(0)
dist = Counter(
    weighted_sample(hotels.copy(), scores.copy(), k=1, replace=True)[0]
    for _ in range(1000)
)
print("  1000-draw distribution (expect Grand Palace ~37%):")
for h in hotels:
    print(f"  {h:<20} {dist[h]:4d}  {'█' * (dist[h]//10)}")

# one-liner numpy equivalent
probs    = np.array(scores) / sum(scores)
np_samps = np.random.choice(hotels, size=5, p=probs, replace=False)
print(f"\n  numpy one-liner (k=5): {list(np_samps)}")


# ══════════════════════════════════════════════════════════════
# SECTION 6: VECTOR OPERATIONS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 6: Vector Operations")
print("=" * 60)

def dot_product(a, b):
    """Σ aᵢbᵢ — sum of element-wise products. O(n)."""
    return sum(x*y for x,y in zip(a,b))

def l2_norm(v):
    """||v||₂ = sqrt(Σ vᵢ²) — Euclidean length of vector."""
    return math.sqrt(sum(x**2 for x in v))

def cosine_similarity(a, b):
    """(a·b) / (||a|| × ||b||) → [-1, 1]. 1=identical direction, 0=orthogonal."""
    denom = l2_norm(a) * l2_norm(b)
    return dot_product(a,b) / denom if denom > 0 else 0.0

def normalise(v):
    """v / ||v|| → unit vector (length 1). Cosine sim of normalised = dot product."""
    n = l2_norm(v)
    return [x/n for x in v] if n > 0 else v

a, b = [1,2,3], [4,5,6]
print(f"  a={a}  b={b}")
print(f"  dot:      {dot_product(a,b)}   (numpy: {np.dot(a,b)})")
print(f"  l2_norm:  {l2_norm(a):.4f}  (numpy: {np.linalg.norm(a):.4f})")
print(f"  cosine:   {cosine_similarity(a,b):.4f}  (numpy: {np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)):.4f})")
print(f"  normalise: {[round(x,4) for x in normalise(a)]}  ||→||={l2_norm(normalise(a)):.4f}")


# ══════════════════════════════════════════════════════════════
# SECTION 7: FORMULA CHEAT SHEET
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 7: Quick Reference")
print("=" * 60)
print("""
  METRIC       FORMULA                    NOTES
  ──────────────────────────────────────────────────────────────
  RMSE         sqrt(mean((y-ŷ)²))        penalises large errors
  MAE          mean(|y-ŷ|)               robust, interpretable $
  MAPE         mean(|y-ŷ|/|y|)×100      best for stakeholders
  R²           1 - SS_res/SS_tot         1=perfect, 0=predict mean

  Precision    TP/(TP+FP)                of predicted +, % correct
  Recall       TP/(TP+FN)                of actual +, % caught
  F1           2PR/(P+R)                 harmonic: both must be high
  Accuracy     (TP+TN)/N                 MISLEADING if imbalanced

  Sigmoid      1/(1+e^-x)               (0,1) — binary output
  Softmax      e^xᵢ/Σe^xⱼ              sums to 1 — multi-class
  ReLU         max(0,x)                  no vanishing gradient

  Variance     Σ(x-μ)²/(n-1)            Bessel's: divide by n-1
  Cosine sim   (a·b)/(||a||·||b||)      [-1,1]; 1=same direction
""")
print("All sections complete.")
