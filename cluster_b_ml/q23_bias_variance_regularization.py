"""
Q23 — Bias-Variance Tradeoff + Regularization [GUARANTEED THEORY QUESTION]
Target time: 15 min | Requires: numpy, sklearn, matplotlib

APPROACH (say this in the first 30 seconds):
"Bias-variance is the fundamental decomposition of generalisation error.
Bias is how wrong the model is on average — underfitting. Variance is
how much predictions change with different training sets — overfitting.
Total error = Bias² + Variance + Irreducible noise. You can't eliminate
all three simultaneously — reducing bias usually increases variance and
vice versa. Regularization is the main tool for managing variance."

THE CORE MATH:
  E[(y - ŷ)²] = Bias² + Variance + σ²
  Bias²     = (E[ŷ] - f(x))²     ← systematic error, model too simple
  Variance  = E[(ŷ - E[ŷ])²]     ← sensitivity to training data, model too complex
  σ²        = irreducible noise   ← can't fix this

THE 60-SECOND ANSWER FOR AIMÉ:
  "High bias → model too simple, misses the pattern. High variance → model
  memorises training data, fails on new data. You diagnose with learning curves:
  if both train and val error are high → bias. If train is low but val is high →
  variance. Regularization (L1, L2, dropout, early stopping) adds a penalty that
  shrinks model complexity, trading a bit of bias for a large reduction in variance.
  The sweet spot is where val error is minimised."
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ─────────────────────────────────────────────────────────────
# SECTION 1: Visualise bias-variance via polynomial complexity
# ─────────────────────────────────────────────────────────────

def make_data(n=100, noise=0.3):
    """True function: y = sin(x) + noise."""
    X = np.sort(np.random.uniform(0, 2 * np.pi, n))
    y = np.sin(X) + np.random.normal(0, noise, n)
    return X.reshape(-1, 1), y


def fit_polynomial(X_train, y_train, X_test, y_test, degree):
    """Fit polynomial regression of given degree, return train/test RMSE."""
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("reg",  LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, pipe.predict(X_train)))
    test_rmse  = np.sqrt(mean_squared_error(y_test,  pipe.predict(X_test)))
    return train_rmse, test_rmse, pipe


print("=" * 60)
print("SECTION 1: Bias-Variance via Polynomial Degree")
print("=" * 60)

X, y = make_data(n=120)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

degrees    = [1, 2, 4, 8, 15, 20]
train_errs = []
test_errs  = []

print(f"\n{'Degree':>8} {'Train RMSE':>12} {'Test RMSE':>12} {'Diagnosis':>20}")
print("-" * 56)

for d in degrees:
    tr, te, _ = fit_polynomial(X_train, y_train, X_test, y_test, d)
    train_errs.append(tr)
    test_errs.append(te)
    if tr > 0.25 and te > 0.25:
        diag = "High Bias (underfit)"
    elif te > tr * 2:
        diag = "High Variance (overfit)"
    else:
        diag = "Good Fit ✓"
    print(f"{d:>8} {tr:>12.4f} {te:>12.4f} {diag:>20}")

print("""
Key insight:
  Degree 1  → both errors high → HIGH BIAS (line can't fit sine wave)
  Degree 4  → low train, low test → GOOD FIT (captures the curve)
  Degree 20 → very low train, high test → HIGH VARIANCE (memorises noise)
""")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Learning curves — the diagnostic tool
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 2: Learning Curves — How to Diagnose Live")
print("=" * 60)

print("""
HOW TO READ LEARNING CURVES (say this to Aimé):

  HIGH BIAS pattern:
    Train loss:  high and flat           ────────────────
    Val loss:    high and converges to   ─────────────────
    Gap:         small (both bad)
    Fix:         more complex model, more features, less regularization

  HIGH VARIANCE pattern:
    Train loss:  low                     _______________
    Val loss:    much higher, converges  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    Gap:         large
    Fix:         more data, dropout, L2 regularization, simpler model

  GOOD FIT pattern:
    Train loss:  low
    Val loss:    slightly higher but close to train
    Gap:         small
""")

# Compute actual learning curves for degree-1 (bias) vs degree-15 (variance)
from sklearn.pipeline import Pipeline as SKPipeline

def get_learning_curve(degree, X, y):
    pipe = SKPipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("reg",  LinearRegression()),
    ])
    sizes, train_scores, val_scores = learning_curve(
        pipe, X, y,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    return sizes, -train_scores.mean(axis=1), -val_scores.mean(axis=1)

sizes_bias, tr_bias, val_bias     = get_learning_curve(1,  X, y)
sizes_var,  tr_var,  val_var      = get_learning_curve(15, X, y)
sizes_good, tr_good, val_good     = get_learning_curve(4,  X, y)

print(f"{'Model':>12} {'Train MSE':>12} {'Val MSE':>12} {'Gap':>10}")
print("-" * 48)
print(f"{'Degree-1':>12} {tr_bias[-1]:>12.4f} {val_bias[-1]:>12.4f} {val_bias[-1]-tr_bias[-1]:>10.4f}  ← HIGH BIAS")
print(f"{'Degree-4':>12} {tr_good[-1]:>12.4f} {val_good[-1]:>12.4f} {val_good[-1]-tr_good[-1]:>10.4f}  ← GOOD FIT")
print(f"{'Degree-15':>12} {tr_var[-1]:>12.4f} {val_var[-1]:>12.4f} {val_var[-1]-tr_var[-1]:>10.4f}  ← HIGH VARIANCE")


# ─────────────────────────────────────────────────────────────
# SECTION 3: L1 vs L2 vs ElasticNet — side-by-side
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: L1 vs L2 vs ElasticNet Regularization")
print("=" * 60)

print("""
REGULARIZATION — THE 60-SECOND ANSWER:

  L2 (Ridge):    Loss = MSE + λ Σ wᵢ²
    → Shrinks all weights toward zero but never exactly zero
    → Good when all features matter (multicollinearity)
    → Bayesian: Gaussian prior on weights
    → Closed-form solution: (XᵀX + λI)⁻¹ Xᵀy

  L1 (Lasso):    Loss = MSE + λ Σ |wᵢ|
    → Drives some weights to EXACTLY zero → automatic feature selection
    → Good for sparse, high-dim data (many irrelevant features)
    → Bayesian: Laplace prior on weights
    → No closed-form; needs coordinate descent / proximal gradient

  ElasticNet:    Loss = MSE + λ₁ Σ |wᵢ| + λ₂ Σ wᵢ²
    → Combines both: some sparsity + handles correlated features
    → Use when you suspect groups of correlated features

  Dropout (NNs): randomly zero p% of neurons each forward pass
    → Ensemble effect: each pass trains a different sub-network
    → Forces redundant representations → generalization

  WHY L1 GIVES SPARSITY (the geometric intuition):
    L2 penalty is a circle (smooth, gradient never hits axes exactly)
    L1 penalty is a diamond (corners on the axes → optimal solution
    is often exactly at a corner → weight = 0)
""")

# Fit all three on high-dimensional noisy data
n_samples, n_features = 200, 50   # more features than informative ones
X_reg = np.random.randn(n_samples, n_features)
true_coef = np.zeros(n_features)
true_coef[:10] = np.random.randn(10) * 3   # only 10/50 features matter
y_reg = X_reg @ true_coef + np.random.randn(n_samples) * 0.5

X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.25, random_state=42)

models = {
    "Ridge (L2)":        Ridge(alpha=1.0),
    "Lasso (L1)":        Lasso(alpha=0.1, max_iter=5000),
    "ElasticNet (L1+L2)": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
}

print(f"\n{'Model':>20} {'Test RMSE':>12} {'Zero weights':>14} {'Non-zero':>10}")
print("-" * 60)
for name, model in models.items():
    model.fit(X_tr, y_tr)
    rmse     = np.sqrt(mean_squared_error(y_te, model.predict(X_te)))
    zeros    = (np.abs(model.coef_) < 1e-6).sum()
    nonzeros = (np.abs(model.coef_) >= 1e-6).sum()
    print(f"{name:>20} {rmse:>12.4f} {zeros:>14} {nonzeros:>10}")

print(f"\n  True: only 10 out of 50 features matter.")
print(f"  Lasso finds near-sparse solution — closest to ground truth.")
print(f"  Ridge keeps all 50 weights non-zero.")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Lambda sweep — bias-variance trade-off curve
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 4: Lambda Sweep — Finding the Sweet Spot")
print("=" * 60)

alphas = np.logspace(-3, 3, 30)
ridge_train, ridge_val = [], []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_tr, y_tr)
    ridge_train.append(np.sqrt(mean_squared_error(y_tr, ridge.predict(X_tr))))
    ridge_val.append(np.sqrt(mean_squared_error(y_te, ridge.predict(X_te))))

best_alpha = alphas[np.argmin(ridge_val)]
best_val   = min(ridge_val)

print(f"\n  λ too small (near 0): val RMSE = {ridge_val[0]:.4f}  ← high variance")
print(f"  λ optimal ({best_alpha:.3f}):     val RMSE = {best_val:.4f}  ← sweet spot ✓")
print(f"  λ too large (1000):  val RMSE = {ridge_val[-1]:.4f}  ← high bias")
print("""
  HOW TO FIND λ IN PRACTICE:
    1. Grid search with cross-validation (RidgeCV, LassoCV)
    2. Plot val error vs log(λ) — pick the elbow/minimum
    3. Rule of thumb: start at λ=1, sweep log-space ±3 decades
""")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Dropout in Neural Networks
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 5: Dropout — Regularization for NNs")
print("=" * 60)

import torch
import torch.nn as nn

class NetWithDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(p / 2),
            nn.Linear(32, 1),
        )
    def forward(self, x): return self.net(x)

model_drop = NetWithDropout(p=0.3)

x_sample = torch.randn(4, 10)

# Training mode: dropout is ACTIVE
model_drop.train()
out_train_1 = model_drop(x_sample)
out_train_2 = model_drop(x_sample)

# Eval mode: dropout is DISABLED (deterministic)
model_drop.eval()
out_eval_1 = model_drop(x_sample)
out_eval_2 = model_drop(x_sample)

print(f"\n  Training outputs differ (dropout active):")
print(f"    Run 1: {out_train_1.detach().numpy().flatten().round(3)}")
print(f"    Run 2: {out_train_2.detach().numpy().flatten().round(3)}")
print(f"\n  Eval outputs identical (dropout off):")
print(f"    Run 1: {out_eval_1.detach().numpy().flatten().round(3)}")
print(f"    Run 2: {out_eval_2.detach().numpy().flatten().round(3)}")

print("""
  KEY DROPOUT FACTS:
    model.train() → dropout active, outputs stochastic
    model.eval()  → dropout disabled, outputs deterministic
    PyTorch scales outputs by 1/(1-p) during training so
    expected values match at eval — no rescaling needed at test time.

  WHY IT WORKS:
    Each forward pass uses a different sub-network → ensemble effect
    Forces neurons to learn independently → redundant representations
    Reduces co-adaptation (neurons relying on specific others)
""")


# ─────────────────────────────────────────────────────────────
# SECTION 6: Summary table — the answer on one slide
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 6: Summary — The Cheat Sheet")
print("=" * 60)
print("""
  SYMPTOM             DIAGNOSIS        FIX
  ─────────────────────────────────────────────────────────────────
  Train ↑  Val ↑      High Bias        More complex model
  (both high)         (underfitting)   More features
                                       Less regularization
                                       Train longer

  Train ↓  Val ↑      High Variance    More training data
  (large gap)         (overfitting)    L1/L2 regularization
                                       Dropout (NNs)
                                       Early stopping
                                       Simpler model / fewer features
                                       Cross-validation

  Train ↓  Val ↓      Good fit ✓       Ship it. Monitor drift.
  (small gap)

  REGULARIZER    PENALTY         EFFECT              WHEN TO USE
  ──────────────────────────────────────────────────────────────────
  L2 (Ridge)     λ Σ wᵢ²         Shrinks, no zeros    Correlated features
  L1 (Lasso)     λ Σ|wᵢ|         Shrinks to zero      Feature selection
  ElasticNet     λ₁|w| + λ₂w²    Both effects         Both problems
  Dropout        random zeros     Ensemble effect      NNs
  Early stopping stops at min val  Implicit reg         NNs, GBMs
""")
print("All sections complete.")
