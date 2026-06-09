"""
Q27 — End-to-End ML Model [VERY LIKELY LIVE TASK]
Target time: 20 min | Requires: pandas, sklearn, lightgbm

APPROACH (say this first 60 seconds):
"I'll follow the standard ML pipeline: load → EDA → clean → feature
engineering → train/test split → baseline (logistic regression) →
improved model (LightGBM) → evaluate with the RIGHT metric for the task
→ discuss overfitting and what I'd do next. I'll start with the simplest
model that could work, not the fanciest. A clean LightGBM baseline with
good evaluation beats a sloppy neural net every time."

THE GOLDEN RULE — say this out loud before coding:
  'Before I pick a model, let me pick the metric.
   The target is imbalanced (15% cancellation), so accuracy is misleading.
   I'll use PR-AUC as primary (emphasises the minority class) and
   ROC-AUC as secondary. I'll also report precision/recall/F1 on a
   threshold tuned for the business use-case.'

REPORTED TASKS:
  - predict hotel booking cancellation
  - predict flight delay
  - predict customer churn
  → All are binary classification with imbalanced targets
  → Same pipeline works for all
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ─────────────────────────────────────────────────────────────
# SECTION 1: Generate dataset + EDA
# ─────────────────────────────────────────────────────────────

def make_cancellation_dataset(n=3000):
    """Simulate hotel booking cancellation dataset."""
    lead_days    = np.random.randint(0, 180, n)
    nights       = np.random.randint(1, 14, n)
    price        = np.random.normal(200, 80, n).clip(30, 600)
    room_type    = np.random.choice(["standard","deluxe","suite"], n, p=[0.5,0.35,0.15])
    channel      = np.random.choice(["direct","ota","phone","corporate"], n, p=[0.3,0.45,0.1,0.15])
    is_repeat    = np.random.randint(0, 2, n)
    party_size   = np.random.randint(1, 6, n)
    month        = np.random.randint(1, 13, n)
    has_requests = np.random.randint(0, 2, n)

    # Cancellation logic: long lead time, OTA, no repeat → higher cancel rate
    logit = (
        -2.5
        + 0.015 * lead_days
        + 0.3   * (channel == "ota").astype(float)
        - 0.8   * is_repeat
        + 0.001 * price
        + 0.1   * has_requests
        + np.random.normal(0, 0.5, n)
    )
    prob_cancel = 1 / (1 + np.exp(-logit))
    cancelled   = (np.random.rand(n) < prob_cancel).astype(int)

    return pd.DataFrame({
        "lead_days": lead_days, "nights": nights, "price": price,
        "room_type": room_type, "channel": channel,
        "is_repeat": is_repeat, "party_size": party_size,
        "month": month, "has_requests": has_requests,
        "cancelled": cancelled,
    })


print("=" * 60)
print("SECTION 1: Load + EDA")
print("=" * 60)

df = make_cancellation_dataset()
print(f"\nShape: {df.shape}")
print(f"\nTarget distribution:")
print(df["cancelled"].value_counts(normalize=True).round(3))
print(f"\n→ Imbalanced ({df['cancelled'].mean():.1%} cancel rate)")
print(f"→ Accuracy is MISLEADING here — a model predicting all-0 gets {1-df['cancelled'].mean():.1%}")
print(f"→ Use PR-AUC (minority class) and ROC-AUC\n")
print(df.describe().round(2))


# ─────────────────────────────────────────────────────────────
# SECTION 2: Feature engineering
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2: Feature Engineering")
print("=" * 60)

df_feat = df.copy()

# Numeric features
df_feat["log_lead_days"]  = np.log1p(df_feat["lead_days"])   # right-skewed
df_feat["revenue"]        = df_feat["price"] * df_feat["nights"]
df_feat["price_per_night"] = df_feat["price"]
df_feat["is_peak_season"] = df_feat["month"].isin([6,7,8,12]).astype(int)
df_feat["is_weekend_dep"] = (df_feat["month"] % 2 == 0).astype(int)  # proxy

# Encode categoricals
for col in ["room_type", "channel"]:
    df_feat = pd.get_dummies(df_feat, columns=[col], drop_first=False)

feature_cols = [c for c in df_feat.columns if c != "cancelled"]
X = df_feat[feature_cols]
y = df_feat["cancelled"]

print(f"Features ({len(feature_cols)}): {list(feature_cols)}")

# Train / val / test split — 70/15/15
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15,
                                               stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.176,
                                                   stratify=y_tv, random_state=42)

print(f"\nSplit: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
print(f"Cancel rate — train:{y_train.mean():.3f}, val:{y_val.mean():.3f}, test:{y_test.mean():.3f}")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Baseline — Logistic Regression
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: Baseline — Logistic Regression")
print("=" * 60)
print("WHY: 'I always start with the simplest model that could work.")
print("     LR is interpretable, fast, and tells me if features have signal.")
print("     If LR is good, I don't need complexity.'")

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    # class_weight="balanced" → penalises minority class misses more
])

lr_pipe.fit(X_train, y_train)
lr_proba = lr_pipe.predict_proba(X_val)[:, 1]
lr_pred  = (lr_proba >= 0.5).astype(int)

lr_roc   = roc_auc_score(y_val, lr_proba)
lr_pr    = average_precision_score(y_val, lr_proba)

print(f"\n  ROC-AUC:  {lr_roc:.4f}")
print(f"  PR-AUC:   {lr_pr:.4f}")
print(f"\n{classification_report(y_val, lr_pred, target_names=['keep','cancel'])}")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Improved model — LightGBM
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 4: LightGBM — Production-Grade Model")
print("=" * 60)
print("WHY LightGBM over XGBoost for prototyping:")
print("  - Faster training (histogram-based, leaf-wise growth)")
print("  - Handles categoricals natively (no need to one-hot)")
print("  - scale_pos_weight handles imbalance without resampling")

try:
    from lightgbm import LGBMClassifier

    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())

    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        scale_pos_weight=scale_pos,  # imbalance handling
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[],
    )

    lgbm_proba = lgbm.predict_proba(X_val)[:, 1]
    lgbm_pred  = (lgbm_proba >= 0.5).astype(int)

    lgbm_roc = roc_auc_score(y_val, lgbm_proba)
    lgbm_pr  = average_precision_score(y_val, lgbm_proba)

    print(f"\n  ROC-AUC:  {lgbm_roc:.4f}  (vs LR: {lr_roc:.4f})")
    print(f"  PR-AUC:   {lgbm_pr:.4f}  (vs LR: {lr_pr:.4f})")
    print(f"\n{classification_report(y_val, lgbm_pred, target_names=['keep','cancel'])}")

    # Feature importance
    feat_imp = pd.Series(lgbm.feature_importances_, index=feature_cols)
    print("Top 8 features:")
    print(feat_imp.nlargest(8).round(0).to_string())

    best_proba = lgbm_proba
    best_name  = "LightGBM"

except ImportError:
    print("  LightGBM not installed — using Logistic Regression as best model")
    best_proba = lr_proba
    best_name  = "LogisticRegression"


# ─────────────────────────────────────────────────────────────
# SECTION 5: Threshold tuning — the business decision
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5: Threshold Tuning — The Senior Signal")
print("=" * 60)
print("""
DEFAULT threshold of 0.5 is almost never optimal for imbalanced data.

BUSINESS FRAMING (say this to Aimé):
  'The threshold depends on the cost asymmetry:
   - False negative (miss a cancellation): hotel room sits empty, ~$200 lost
   - False positive (flag a booking that stays): unnecessary overbooking action
   If FN cost >> FP cost → lower the threshold to catch more cancellations.
   I'll find the threshold that maximises F1 or a custom cost function.'
""")

precisions, recalls, thresholds = precision_recall_curve(y_val, best_proba)
f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
best_thresh = thresholds[np.argmax(f1_scores)]
best_pred   = (best_proba >= best_thresh).astype(int)

print(f"  Default threshold (0.50) F1: {f1_scores[np.argmin(np.abs(thresholds - 0.5))]:.4f}")
print(f"  Optimal threshold ({best_thresh:.2f}) F1: {f1_scores.max():.4f}")
print(f"\n  At optimal threshold:\n"
      f"{classification_report(y_val, best_pred, target_names=['keep','cancel'])}")


# ─────────────────────────────────────────────────────────────
# SECTION 6: Final evaluation on held-out test set
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 6: Held-Out Test Set — Final Numbers")
print("=" * 60)
print("IMPORTANT: evaluate on test set ONCE, at the end.")
print("'If I touch the test set during development I'm leaking information.'")

if best_name == "LightGBM":
    test_proba = lgbm.predict_proba(X_test)[:, 1]
else:
    test_proba = lr_pipe.predict_proba(X_test)[:, 1]

test_pred = (test_proba >= best_thresh).astype(int)
test_roc  = roc_auc_score(y_test, test_proba)
test_pr   = average_precision_score(y_test, test_proba)

print(f"\n  Model: {best_name}")
print(f"  Test ROC-AUC: {test_roc:.4f}")
print(f"  Test PR-AUC:  {test_pr:.4f}")
print(f"\n{classification_report(y_test, test_pred, target_names=['keep','cancel'])}")

cm = confusion_matrix(y_test, test_pred)
print(f"  Confusion matrix (rows=actual, cols=predicted):")
print(f"              Pred keep  Pred cancel")
print(f"  Actual keep    {cm[0,0]:6d}      {cm[0,1]:6d}")
print(f"  Actual cancel  {cm[1,0]:6d}      {cm[1,1]:6d}")


# ─────────────────────────────────────────────────────────────
# SECTION 7: Production talking points
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 7: What I'd Do Next (Production Angle)")
print("=" * 60)
print("""
Aimé WILL ask: "How would you deploy and monitor this?"

1. CALIBRATION
   Probabilities from LightGBM are not well-calibrated.
   Apply Platt scaling or isotonic regression after training:
     from sklearn.calibration import CalibratedClassifierCV

2. MONITORING
   Track: distribution of predicted probabilities (PSI)
          actual cancellation rate on recent bookings
          model ROC-AUC on weekly holdout
   Alert if: PSI > 0.25 OR ROC-AUC drops > 5%

3. RETRAINING STRATEGY
   Scheduled: weekly retraining on a rolling 90-day window
   Triggered: when drift is detected

4. FAIRNESS / SEGMENTATION
   Does the model perform equally across channels and room types?
   Check ROC-AUC per segment — a model that's great overall but
   terrible for a specific segment has a hidden problem.

5. FEATURE STORE
   lead_days, price, is_repeat → pre-compute and cache per booking
   Serve from feature store at inference time (<10ms SLA)

6. SCALE
   3k rows: sklearn is fine
   30M rows: Spark + LightGBM distributed, or train on sample + serve sklearn
""")
print("All sections complete.")
