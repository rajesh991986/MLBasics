"""
Q28 — From-Scratch Functions [SOMETIMES ASKED]
Target time: 15 min | Requires: numpy only (stdlib-safe alternatives included)

APPROACH (say first 30 seconds):
"I'll implement each function from first principles, then verify against
numpy/sklearn. For each one I'll state the formula, edge cases, and
time complexity before writing code."

REPORTED TASKS:
  - calculate_rmse / calculate_mae
  - sample variance (with Bessel's correction)
  - weighted random sampler
  - precision / recall / F1 from scratch
  - softmax from scratch
  - sigmoid from scratch
  - normalise a vector

INTERVIEW SIGNAL:
  Getting these right fast shows you understand the math, not just the API.
  Interviewers use these as a warm-up before the main task.
"""

import numpy as np
import math
from collections import Counter

# ─────────────────────────────────────────────────────────────
# SECTION 1: Regression metrics
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 1: Regression Metrics from Scratch")
print("=" * 60)


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.
    Formula: sqrt( mean( (y_true - y_pred)^2 ) )
    Penalises large errors heavily (squares them).
    NOT robust to outliers — use MAE if you want that.
    Time: O(n) | Space: O(1)
    """
    n = len(y_true)
    ss = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    return math.sqrt(ss / n)


def mae(y_true, y_pred):
    """
    Mean Absolute Error.
    Formula: mean( |y_true - y_pred| )
    Robust to outliers. Interpretable: 'average dollar error'.
    Time: O(n) | Space: O(1)
    """
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def mape(y_true, y_pred, eps=1e-8):
    """
    Mean Absolute Percentage Error.
    Formula: mean( |y_true - y_pred| / |y_true| ) * 100
    Interpretable: 'off by X% on average'.
    Undefined when y_true = 0 → guard with eps.
    Time: O(n) | Space: O(1)
    """
    return 100 * sum(
        abs(t - p) / (abs(t) + eps) for t, p in zip(y_true, y_pred)
    ) / len(y_true)


def r_squared(y_true, y_pred):
    """
    R² (coefficient of determination).
    Formula: 1 - SS_res / SS_tot
    1.0 = perfect fit | 0.0 = predicts mean | <0 = worse than mean
    """
    mean_y  = sum(y_true) / len(y_true)
    ss_tot  = sum((t - mean_y) ** 2 for t in y_true)
    ss_res  = sum((t - p)      ** 2 for t, p in zip(y_true, y_pred))
    return 1 - ss_res / (ss_tot + 1e-9)


# Test
y_true = [3.0, 4.5, 2.1, 5.8, 3.3]
y_pred = [2.8, 4.7, 2.5, 5.2, 3.0]

print(f"  RMSE:  {rmse(y_true, y_pred):.4f}")
print(f"  MAE:   {mae(y_true, y_pred):.4f}")
print(f"  MAPE:  {mape(y_true, y_pred):.4f}%")
print(f"  R²:    {r_squared(y_true, y_pred):.4f}")

# Verify against numpy
print(f"\n  NumPy RMSE: {np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2)):.4f}  ✓")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Statistics
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2: Statistics from Scratch")
print("=" * 60)


def sample_variance(data):
    """
    Sample variance with Bessel's correction (divide by n-1).
    WHY n-1: we're estimating population variance from a sample.
             Dividing by n underestimates it. n-1 corrects the bias.
             This is 'degrees of freedom' — we lose 1 because we
             estimated the mean from the same data.
    Formula: sum( (x - mean)^2 ) / (n-1)
    """
    n    = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / (n - 1)


def sample_std(data):
    return math.sqrt(sample_variance(data))


def median(data):
    """O(n log n) — sort then pick middle."""
    s = sorted(data)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def mode(data):
    """Return most frequent value. O(n)."""
    return Counter(data).most_common(1)[0][0]


def percentile(data, p):
    """Linear interpolation percentile. O(n log n)."""
    s = sorted(data)
    n = len(s)
    idx = (n - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


data = [4, 7, 13, 2, 8, 1, 9, 4, 7, 4, 3, 12]
print(f"  Data: {sorted(data)}")
print(f"  Mean:     {sum(data)/len(data):.4f}  (numpy: {np.mean(data):.4f})")
print(f"  Variance: {sample_variance(data):.4f} (numpy: {np.var(data, ddof=1):.4f})")
print(f"  Std:      {sample_std(data):.4f}  (numpy: {np.std(data, ddof=1):.4f})")
print(f"  Median:   {median(data):.1f}   (numpy: {np.median(data):.1f})")
print(f"  Mode:     {mode(data)}          (most frequent: 4)")
print(f"  P75:      {percentile(data, 75):.2f} (numpy: {np.percentile(data, 75):.2f})")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Classification metrics
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: Classification Metrics from Scratch")
print("=" * 60)


def precision_recall_f1(y_true, y_pred, pos_label=1):
    """
    Precision = TP / (TP + FP)  — of predicted positives, how many are real?
    Recall    = TP / (TP + FN)  — of real positives, how many did we catch?
    F1        = 2 * P * R / (P + R)  — harmonic mean

    WHY harmonic mean not arithmetic?
      Arithmetic mean rewards getting one very high.
      Harmonic mean is only high when BOTH are high.
      e.g. P=1.0, R=0.01 → arithmetic=0.505, harmonic=0.02
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p == pos_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != pos_label and p == pos_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p != pos_label)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)


y_true_cls = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred_cls = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

p, r, f = precision_recall_f1(y_true_cls, y_pred_cls)
print(f"  Precision: {p}")
print(f"  Recall:    {r}")
print(f"  F1:        {f}")

from sklearn.metrics import precision_score, recall_score, f1_score
print(f"\n  sklearn P/R/F1: {precision_score(y_true_cls,y_pred_cls):.4f} / "
      f"{recall_score(y_true_cls,y_pred_cls):.4f} / "
      f"{f1_score(y_true_cls,y_pred_cls):.4f}  ✓")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Neural net building blocks
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 4: NN Building Blocks from Scratch")
print("=" * 60)


def sigmoid(x):
    """σ(x) = 1 / (1 + e^-x). Maps any real → (0,1). Used in binary output layer."""
    return 1 / (1 + math.exp(-x))


def sigmoid_vec(x):
    """Vectorised sigmoid — numerically stable (avoids exp overflow)."""
    x = np.array(x, dtype=float)
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def softmax(logits):
    """
    Softmax: converts raw scores → probability distribution.
    Formula: e^xᵢ / Σ e^xⱼ
    Numerically stable trick: subtract max before exp (avoids overflow).
    Output sums to 1, all values in (0,1).
    """
    logits = np.array(logits, dtype=float)
    shifted = logits - logits.max()      # stability: max → 0 after shift
    exps    = np.exp(shifted)
    return exps / exps.sum()


def relu(x):
    """ReLU = max(0, x). Zero for negatives, identity for positives."""
    return max(0.0, x)


def relu_vec(x):
    return np.maximum(0, np.array(x, dtype=float))


# Tests
print(f"  sigmoid(0)   = {sigmoid(0):.4f}  (expect 0.5)")
print(f"  sigmoid(2)   = {sigmoid(2):.4f}  (expect 0.8808)")
print(f"  sigmoid(-10) = {sigmoid(-10):.6f} (expect ~0)")

print(f"\n  softmax([1,2,3]) = {softmax([1,2,3]).round(4)}")
print(f"  sum = {softmax([1,2,3]).sum():.4f}  (expect 1.0)")
print(f"  softmax([1000,1001,1002]) = {softmax([1000,1001,1002]).round(4)}  (stable)")

print(f"\n  relu(-3) = {relu(-3)}   relu(5) = {relu(5)}")
print(f"  relu_vec([-2,-1,0,1,2]) = {relu_vec([-2,-1,0,1,2])}")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Weighted random sampler
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5: Weighted Random Sampler")
print("=" * 60)
print("""
USE CASE: sample from imbalanced dataset where rare classes should
appear more often. E.g. pick a hotel to recommend with probability
proportional to its popularity score.

ALGORITHM (alias method in one line with numpy):
  np.random.choice(items, size=k, p=normalised_weights, replace=False)

FROM SCRATCH (cumulative sum + binary search):
  1. Normalise weights to sum to 1 (probability distribution)
  2. Compute cumulative sum
  3. For each sample: generate U ~ Uniform(0,1), find first bin where cum_sum > U
""")


def weighted_sample(items, weights, k, replace=False):
    """
    Sample k items with given weights (without replacement by default).
    Time: O(k log n) with binary search | Space: O(n)
    """
    weights = [w / sum(weights) for w in weights]   # normalise
    cum     = []
    running = 0.0
    for w in weights:
        running += w
        cum.append(running)

    selected = []
    available = list(range(len(items)))

    for _ in range(k):
        u = np.random.random()
        # Binary search for insertion point
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        idx = available[lo] if replace else lo
        selected.append(items[idx] if replace else items[lo])
        if not replace:
            items   = [x for i, x in enumerate(items) if i != lo]
            weights = [w for i, w in enumerate(weights) if i != lo]
            cum     = []
            running = 0.0
            s       = sum(weights)
            for w in weights:
                running += w / s if s > 0 else 0
                cum.append(running)
    return selected


hotels  = ["Budget Inn", "City Hotel", "Grand Palace", "Boutique Stay", "Airport Lodge"]
scores  = [2, 5, 10, 7, 3]   # popularity scores

np.random.seed(0)
samples = [weighted_sample(hotels.copy(), scores.copy(), k=1, replace=True)[0]
           for _ in range(1000)]
dist = Counter(samples)
print("  Sample distribution from 1000 draws:")
for h in hotels:
    bar = "█" * int(dist[h] / 10)
    print(f"  {h:<20} {dist[h]:4d} {bar}")

# Numpy equivalent (one-liner)
probs   = np.array(scores) / sum(scores)
np_samp = np.random.choice(hotels, size=5, p=probs, replace=False)
print(f"\n  numpy equivalent (k=10): {list(np_samp)}")


# ─────────────────────────────────────────────────────────────
# SECTION 6: Vector operations
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 6: Vector Operations from Scratch")
print("=" * 60)


def dot_product(a, b):
    """O(n). Must have same length."""
    return sum(x * y for x, y in zip(a, b))


def l2_norm(v):
    """||v||₂ = sqrt(Σ vᵢ²)"""
    return math.sqrt(sum(x ** 2 for x in v))


def cosine_similarity(a, b):
    """cos(θ) = (a · b) / (||a|| * ||b||). Range [-1, 1]."""
    denom = l2_norm(a) * l2_norm(b)
    if denom == 0:
        return 0.0
    return dot_product(a, b) / denom


def normalise(v):
    """Unit vector: v / ||v||. Makes cosine sim = dot product."""
    n = l2_norm(v)
    return [x / n for x in v] if n > 0 else v


a = [1, 2, 3]
b = [4, 5, 6]
print(f"  a={a}, b={b}")
print(f"  dot(a,b)     = {dot_product(a,b)}     (numpy: {np.dot(a,b)})")
print(f"  l2_norm(a)   = {l2_norm(a):.4f}  (numpy: {np.linalg.norm(a):.4f})")
print(f"  cosine(a,b)  = {cosine_similarity(a,b):.4f}  "
      f"(numpy: {np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)):.4f})")
print(f"  normalise(a) = {[round(x,4) for x in normalise(a)]}")
print(f"  ||normalised||= {l2_norm(normalise(a)):.4f}  (expect 1.0)")


# ─────────────────────────────────────────────────────────────
# SECTION 7: Quick reference card
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 7: Quick Reference — Formulas at a Glance")
print("=" * 60)
print("""
  METRIC      FORMULA                          NOTES
  ──────────────────────────────────────────────────────────────
  RMSE        sqrt(mean((y-ŷ)²))              Penalises large errors
  MAE         mean(|y-ŷ|)                     Robust to outliers
  MAPE        mean(|y-ŷ|/|y|) * 100%         Interpretable %
  R²          1 - SS_res/SS_tot               1=perfect, 0=mean

  Precision   TP/(TP+FP)                      Of predicted +, how many real?
  Recall      TP/(TP+FN)                      Of real +, how many caught?
  F1          2*P*R/(P+R)                     Harmonic mean (both must be high)
  Accuracy    (TP+TN)/N                       Misleading if imbalanced!

  Sigmoid     1/(1+e^-x)                      Output ∈ (0,1), binary output
  Softmax     e^xᵢ / Σe^xⱼ                  Output sums to 1, multi-class
  ReLU        max(0, x)                       Non-linearity, no vanishing grad

  Variance    Σ(x-μ)²/(n-1)                  Bessel's correction for sample
  Std Dev     sqrt(variance)                  Same units as data
  Cosine Sim  (a·b)/(||a||·||b||)            Range [-1,1]; 1=identical
""")
print("All sections complete.")
