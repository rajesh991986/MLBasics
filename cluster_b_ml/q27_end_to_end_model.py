"""
Q27 — End-to-End ML Model [VERY LIKELY LIVE TASK]
Target time: 20 min | Requires: pandas, sklearn, lightgbm

SAY THIS FIRST (60 sec — before touching keyboard):
  "Before I pick a model, I pick the metric.
   Target is imbalanced (15% cancellation) → accuracy is misleading.
   I'll use PR-AUC as primary and ROC-AUC as secondary.
   I'll start with LR as a baseline — if it works, I don't need complexity.
   Then LightGBM. Threshold tuning at the end based on business cost."

PIPELINE ORDER:
  EDA → feature engineering → 70/15/15 split (stratified) →
  LR baseline → LightGBM → threshold tuning → final test eval

WORKS FOR: hotel cancellation, flight delay, churn — same pipeline for all.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


# ── synthetic data: realistic cancellation drivers ─────────────
def make_cancellation_dataset(n=3000):
    lead_days    = np.random.randint(0, 180, n)   # days booked in advance
    nights       = np.random.randint(1, 14, n)
    price        = np.random.normal(200, 80, n).clip(30, 600)
    room_type    = np.random.choice(["standard","deluxe","suite"], n, p=[0.5,0.35,0.15])
    channel      = np.random.choice(["direct","ota","phone","corporate"], n, p=[0.3,0.45,0.1,0.15])
    is_repeat    = np.random.randint(0, 2, n)     # 1 = returning customer
    party_size   = np.random.randint(1, 6, n)
    month        = np.random.randint(1, 13, n)
    has_requests = np.random.randint(0, 2, n)     # special requests → more likely to cancel

    # ground truth: long lead + OTA + first-time → higher cancel rate
    logit = (
        -2.5
        + 0.015 * lead_days                          # longer lead → more likely cancel
        + 0.3   * (channel == "ota").astype(float)   # OTA bookings cancel more
        - 0.8   * is_repeat                          # repeat customers cancel less
        + 0.001 * price
        + 0.1   * has_requests
        + np.random.normal(0, 0.5, n)               # noise
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


# ══════════════════════════════════════════════════════════════
# SECTION 1: LOAD + EDA
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1: Load + EDA")
print("=" * 60)

df = make_cancellation_dataset()
print(f"\nShape: {df.shape}")
print(f"\nTarget distribution:")
print(df["cancelled"].value_counts(normalize=True).round(3))
# KEY POINT: say this out loud
print(f"\n→ {df['cancelled'].mean():.1%} cancel rate — IMBALANCED")
print(f"→ Predicting all-0 gets {1-df['cancelled'].mean():.1%} accuracy — useless baseline")
print(f"→ Use PR-AUC (rewards finding the minority class) + ROC-AUC")
print(f"\n{df.describe().round(2)}")


# ══════════════════════════════════════════════════════════════
# SECTION 2: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: Feature Engineering")
print("=" * 60)

df_feat = df.copy()

# log-transform right-skewed feature (lead_days has a long tail)
df_feat["log_lead_days"]  = np.log1p(df_feat["lead_days"])

# interaction feature — total value of booking
df_feat["revenue"]        = df_feat["price"] * df_feat["nights"]

# binary flags from domain knowledge
df_feat["is_peak_season"] = df_feat["month"].isin([6,7,8,12]).astype(int)
df_feat["is_weekend_dep"] = (df_feat["month"] % 2 == 0).astype(int)  # proxy

# one-hot encode categoricals — no ordinal assumption imposed
for col in ["room_type", "channel"]:
    df_feat = pd.get_dummies(df_feat, columns=[col], drop_first=False)

feature_cols = [c for c in df_feat.columns if c != "cancelled"]
X = df_feat[feature_cols]
y = df_feat["cancelled"]

print(f"Features ({len(feature_cols)}): {list(feature_cols)}")

# stratify=y → preserve class ratio in each split (critical for imbalanced data)
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15,
                                               stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.176,
                                                   stratify=y_tv, random_state=42)
# 0.176 × 0.85 ≈ 0.15 → 70/15/15 split

print(f"\nSplit: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
print(f"Cancel rate — train:{y_train.mean():.3f} val:{y_val.mean():.3f} test:{y_test.mean():.3f}")


# ══════════════════════════════════════════════════════════════
# SECTION 3: BASELINE — LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3: Baseline — Logistic Regression")
print("=" * 60)
print("WHY START HERE: interpretable, fast, tells you if features have signal.")
print("If LR is already good, you don't need complexity.")

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),     # LR needs scaled features; tree models don't
    ("model",  LogisticRegression(
        class_weight="balanced",      # auto-upweight minority class
        max_iter=1000,
        random_state=42
    )),
])

lr_pipe.fit(X_train, y_train)
lr_proba = lr_pipe.predict_proba(X_val)[:, 1]  # probability of cancellation
lr_pred  = (lr_proba >= 0.5).astype(int)        # default threshold

lr_roc = roc_auc_score(y_val, lr_proba)
lr_pr  = average_precision_score(y_val, lr_proba)

print(f"\n  ROC-AUC: {lr_roc:.4f}  |  PR-AUC: {lr_pr:.4f}")
print(f"\n{classification_report(y_val, lr_pred, target_names=['keep','cancel'])}")


# ══════════════════════════════════════════════════════════════
# SECTION 4: LIGHTGBM — PRODUCTION-GRADE MODEL
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 4: LightGBM")
print("=" * 60)
print("WHY LIGHTGBM: faster than XGBoost (leaf-wise growth), handles")
print("categoricals natively, scale_pos_weight for imbalance, no scaling needed.")

try:
    from lightgbm import LGBMClassifier

    # scale_pos_weight = ratio of negatives to positives — upweights minority
    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())

    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,      # small lr → more trees, better generalization
        max_depth=6,
        num_leaves=31,           # controls tree complexity; < 2^max_depth
        scale_pos_weight=scale_pos,
        subsample=0.8,           # row sampling → reduces overfitting
        colsample_bytree=0.8,    # feature sampling → reduces overfitting
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])

    lgbm_proba = lgbm.predict_proba(X_val)[:, 1]
    lgbm_pred  = (lgbm_proba >= 0.5).astype(int)

    lgbm_roc = roc_auc_score(y_val, lgbm_proba)
    lgbm_pr  = average_precision_score(y_val, lgbm_proba)

    print(f"\n  ROC-AUC: {lgbm_roc:.4f} (LR: {lr_roc:.4f})  |  PR-AUC: {lgbm_pr:.4f} (LR: {lr_pr:.4f})")
    print(f"\n{classification_report(y_val, lgbm_pred, target_names=['keep','cancel'])}")

    # feature importance: which features drive the predictions?
    feat_imp = pd.Series(lgbm.feature_importances_, index=feature_cols)
    print("Top 8 features by importance:")
    print(feat_imp.nlargest(8).round(0).to_string())

    best_proba = lgbm_proba
    best_name  = "LightGBM"

except ImportError:
    print("LightGBM not installed — falling back to LR")
    best_proba = lr_proba
    best_name  = "LogisticRegression"


# ══════════════════════════════════════════════════════════════
# SECTION 5: THRESHOLD TUNING — THE SENIOR SIGNAL
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 5: Threshold Tuning")
print("=" * 60)
print("""
WHY: default threshold 0.5 is almost never optimal for imbalanced data.
     The right threshold depends on cost asymmetry:
       FN (miss cancellation): room sits empty → ~$200 lost
       FP (flag a keeper):     unnecessary intervention → small cost
     If FN cost >> FP cost → lower threshold to catch more cancellations.
     Find threshold that maximises F1 (or a custom business metric).
""")

# precision_recall_curve returns P, R at every threshold — sweep to find best F1
precisions, recalls, thresholds = precision_recall_curve(y_val, best_proba)
f1_scores   = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
best_thresh = thresholds[np.argmax(f1_scores)]
best_pred   = (best_proba >= best_thresh).astype(int)

default_f1 = f1_scores[np.argmin(np.abs(thresholds - 0.5))]
print(f"  Default threshold (0.50) F1:            {default_f1:.4f}")
print(f"  Optimal threshold ({best_thresh:.2f}) F1: {f1_scores.max():.4f}")
print(f"\n{classification_report(y_val, best_pred, target_names=['keep','cancel'])}")


# ══════════════════════════════════════════════════════════════
# SECTION 6: FINAL TEST SET EVALUATION
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 6: Held-Out Test Set — Final Numbers")
print("=" * 60)
# RULE: touch the test set ONCE at the very end — not during dev
print("IMPORTANT: evaluate on test set once only. Touching it earlier = leakage.\n")

test_proba = lgbm.predict_proba(X_test)[:, 1] if best_name == "LightGBM" \
             else lr_pipe.predict_proba(X_test)[:, 1]
test_pred  = (test_proba >= best_thresh).astype(int)  # use tuned threshold

test_roc = roc_auc_score(y_test, test_proba)
test_pr  = average_precision_score(y_test, test_proba)

print(f"  Model: {best_name}")
print(f"  Test ROC-AUC: {test_roc:.4f}  |  Test PR-AUC: {test_pr:.4f}")
print(f"\n{classification_report(y_test, test_pred, target_names=['keep','cancel'])}")

cm = confusion_matrix(y_test, test_pred)
print(f"  Confusion matrix (rows=actual, cols=predicted):")
print(f"              Pred keep  Pred cancel")
print(f"  Actual keep   {cm[0,0]:6d}      {cm[0,1]:6d}")
print(f"  Actual cancel {cm[1,0]:6d}      {cm[1,1]:6d}")


# ══════════════════════════════════════════════════════════════
# SECTION 7: PRODUCTION TALKING POINTS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 7: What I'd Do Next")
print("=" * 60)
print("""
Aimé WILL ask: 'How would you deploy and monitor this?'

1. CALIBRATION — LightGBM probabilities are not well-calibrated
   → Platt scaling: CalibratedClassifierCV(lgbm, method='sigmoid')

2. MONITORING — track weekly:
   → PSI on feature distributions (alert if PSI > 0.25)
   → Actual cancellation rate vs predicted
   → ROC-AUC on a rolling holdout window

3. RETRAINING — weekly retrain on rolling 90-day window
   → Triggered early if drift detected

4. FAIRNESS — check ROC-AUC per channel and room_type
   → Model that's 0.85 overall but 0.60 for OTA is hiding a problem

5. SCALE — 3k rows: sklearn fine; 30M rows: Spark + LightGBM distributed
""")
print("All sections complete.")
