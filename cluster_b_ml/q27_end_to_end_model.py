"""
Q27 — End-to-End ML Model [VERY LIKELY LIVE TASK]
Target time: 20 min | Requires: pandas, sklearn, lightgbm

SAY THIS FIRST (60 sec — before touching keyboard):
  "Before I pick a model, I pick the metric.
   Target is imbalanced (~13% cancellation) → accuracy is misleading.
   I'll use PR-AUC as primary and ROC-AUC as secondary.
   I'll start with LR as a fast interpretable baseline, then LightGBM
   to capture non-linear interactions. Threshold tuning at the end
   based on business cost asymmetry."

PIPELINE ORDER:
  EDA → feature engineering → 70/15/15 split (stratified) →
  LR baseline → LightGBM → threshold tuning → final test eval

WORKS FOR: hotel cancellation, flight delay, churn — same pipeline.
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


# ── synthetic data: non-linear boundary → LightGBM beats LR ───
# make_classification with n_clusters_per_class=3 creates non-convex
# class regions — multiple clusters per class means no single linear
# hyperplane can separate them. LightGBM finds the splits; LR can't.
# Features are wrapped in hotel-booking names so the talk-through sounds natural.
from sklearn.datasets import make_classification

def make_cancellation_dataset(n=5000):
    X_raw, y = make_classification(
        n_samples=n,
        n_features=20,
        n_informative=8,         # 8 features actually matter
        n_redundant=4,           # 4 are linear combinations of informative
        n_clusters_per_class=3,  # KEY: 3 clusters/class → non-linear boundary
        weights=[0.87, 0.13],    # 13% positive = realistic cancel rate
        flip_y=0.03,             # 3% label noise (realistic)
        random_state=42,
    )
    # Wrap in domain-friendly names so code-review sounds natural
    col_names = [
        "lead_days_scaled", "price_scaled", "nights_scaled",
        "lead_days_sq",     "price_log",    "nights_log",
        "is_ota",           "is_repeat",    "party_size_scaled",
        "month_sin",        "month_cos",    "has_requests",
        "is_peak",          "revenue_scaled","price_per_night",
        "lead_bucket",      "price_bucket", "noise_1",
        "noise_2",          "noise_3",
    ]
    df = pd.DataFrame(X_raw, columns=col_names)
    df["cancelled"] = y
    return df


# ══════════════════════════════════════════════════════════════
# SECTION 1: LOAD + EDA
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1: Load + EDA")
print("=" * 60)

df = make_cancellation_dataset(n=5000)  # larger n → stable minority-class metrics
print(f"\nShape: {df.shape}")
print(f"\nTarget distribution:")
print(df["cancelled"].value_counts(normalize=True).round(3))
# SAY THIS — before picking model or metric
print(f"\n→ {df['cancelled'].mean():.1%} cancel rate — IMBALANCED")
print(f"→ Predicting all-0 gets {1-df['cancelled'].mean():.1%} accuracy — meaningless baseline")
print(f"→ Use PR-AUC (rewards finding minority class) + ROC-AUC")
print(f"\n{df.describe().round(2)}")


# ══════════════════════════════════════════════════════════════
# SECTION 2: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: Feature Engineering")
print("=" * 60)

df_feat = df.copy()

# Features already numeric (make_classification outputs floats).
# In a real interview with hotel data you would do:
#   log1p(lead_days), price*nights revenue, pd.get_dummies for categoricals.
# Here we show the pattern comment, then use the data as-is.
# INTERVIEW NOTE: always say what you'd do even if the data is pre-processed.
print("Feature engineering steps (applied to raw hotel data in production):")
print("  log1p(lead_days)     → right-skewed, log makes it symmetric")
print("  price × nights       → total booking value")
print("  is_peak_season       → Jun/Jul/Aug/Dec flag")
print("  pd.get_dummies()     → one-hot for room_type, channel")

feature_cols = [c for c in df_feat.columns if c != "cancelled"]
X = df_feat[feature_cols]
y = df_feat["cancelled"]

print(f"\nFeatures ({len(feature_cols)}): first 8: {list(feature_cols[:8])} ...")

# stratify=y → preserve class ratio across all 3 splits
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15,
                                               stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.176,
                                                   stratify=y_tv, random_state=42)
# 0.176 × 0.85 ≈ 0.15 → gives 70 / 15 / 15 split

print(f"\nSplit: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
print(f"Cancel rate — train:{y_train.mean():.3f}  val:{y_val.mean():.3f}  test:{y_test.mean():.3f}")


# ══════════════════════════════════════════════════════════════
# SECTION 3: BASELINE — LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3: Baseline — Logistic Regression")
print("=" * 60)
print("WHY START HERE: interpretable, fast, tells you if features have signal.")
print("If LR is already great, complexity isn't justified.")

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),    # LR needs scaled features; tree models don't
    ("model",  LogisticRegression(
        class_weight="balanced",     # auto-upweight minority class in loss
        max_iter=1000,
        random_state=42,
    )),
])

lr_pipe.fit(X_train, y_train)
lr_proba = lr_pipe.predict_proba(X_val)[:, 1]   # P(cancellation)
lr_pred  = (lr_proba >= 0.5).astype(int)         # default threshold

lr_roc = roc_auc_score(y_val, lr_proba)
lr_pr  = average_precision_score(y_val, lr_proba)

print(f"\n  ROC-AUC: {lr_roc:.4f}  |  PR-AUC: {lr_pr:.4f}")
print(f"\n{classification_report(y_val, lr_pred, target_names=['keep','cancel'])}")


# ══════════════════════════════════════════════════════════════
# SECTION 4: LIGHTGBM — CAPTURES NON-LINEAR RULES
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 4: LightGBM")
print("=" * 60)
print("WHY LGBM BEATS LR HERE: the cancel rules are threshold-based")
print("(e.g. lead_days > 60 AND channel=OTA). LR needs you to manually")
print("engineer every cross-feature; LGBM finds splits automatically.")

try:
    from lightgbm import LGBMClassifier

    # neg/pos ratio — tells LightGBM to upweight the minority class
    # max(1, ...) guards against edge case where positives > negatives
    scale_pos = max(1, int((y_train == 0).sum() / (y_train == 1).sum()))
    print(f"\n  scale_pos_weight = {scale_pos} (neg/pos ratio)")

    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,      # small lr + more trees → better generalisation
        max_depth=6,
        num_leaves=31,           # max 2^6=64 leaves, 31 gives good regularisation
        scale_pos_weight=scale_pos,
        subsample=0.8,           # row subsampling → reduces overfitting
        colsample_bytree=0.8,    # feature subsampling → reduces overfitting
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])

    lgbm_proba = lgbm.predict_proba(X_val)[:, 1]
    lgbm_pred  = (lgbm_proba >= 0.5).astype(int)

    lgbm_roc = roc_auc_score(y_val, lgbm_proba)
    lgbm_pr  = average_precision_score(y_val, lgbm_proba)

    # LightGBM should beat LR here because data has threshold/interaction rules
    print(f"\n  ROC-AUC: {lgbm_roc:.4f}  (LR: {lr_roc:.4f})")
    print(f"  PR-AUC:  {lgbm_pr:.4f}  (LR: {lr_pr:.4f})")
    print(f"\n{classification_report(y_val, lgbm_pred, target_names=['keep','cancel'])}")

    # top features — sanity check that the right signals were learned
    feat_imp = pd.Series(lgbm.feature_importances_, index=feature_cols)
    print("Top 8 features by importance:")
    print(feat_imp.nlargest(8).round(0).to_string())

    best_proba = lgbm_proba
    best_name  = "LightGBM"

except ImportError:
    print("  LightGBM not installed — using LR as best model")
    best_proba = lr_proba
    best_name  = "LogisticRegression"


# ══════════════════════════════════════════════════════════════
# SECTION 5: THRESHOLD TUNING — THE SENIOR SIGNAL
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 5: Threshold Tuning")
print("=" * 60)
print("""
WHY: default threshold=0.5 is almost never right for imbalanced data.
SAY THIS: "Threshold choice depends on cost asymmetry:
  - Miss a cancellation (FN) → room sits empty → ~$200 loss
  - Flag a keeper (FP)       → unnecessary action → small cost
  If FN >> FP: lower threshold to catch more cancellations."
""")

# sweep every threshold the model produces, find best F1
precisions, recalls, thresholds = precision_recall_curve(y_val, best_proba)
f1_scores   = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
best_thresh = thresholds[np.argmax(f1_scores)]
best_pred   = (best_proba >= best_thresh).astype(int)

default_f1 = f1_scores[np.argmin(np.abs(thresholds - 0.5))]
print(f"  Default threshold (0.50) F1:  {default_f1:.4f}")
print(f"  Optimal threshold ({best_thresh:.2f}) F1: {f1_scores.max():.4f}")
print(f"\n{classification_report(y_val, best_pred, target_names=['keep','cancel'])}")


# ══════════════════════════════════════════════════════════════
# SECTION 6: FINAL TEST SET — TOUCH ONCE, AT THE END
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 6: Held-Out Test Set")
print("=" * 60)
# RULE: evaluate on test set ONCE at the very end
# Touching it earlier = data leakage (you optimised for it implicitly)
print("Evaluating on test set once only. Earlier = leakage.\n")

test_proba = lgbm.predict_proba(X_test)[:, 1] if best_name == "LightGBM" \
             else lr_pipe.predict_proba(X_test)[:, 1]
test_pred  = (test_proba >= best_thresh).astype(int)  # use val-tuned threshold

test_roc = roc_auc_score(y_test, test_proba)
test_pr  = average_precision_score(y_test, test_proba)

print(f"  Model:        {best_name}")
print(f"  Test ROC-AUC: {test_roc:.4f}  |  Test PR-AUC: {test_pr:.4f}")
print(f"\n{classification_report(y_test, test_pred, target_names=['keep','cancel'])}")

cm = confusion_matrix(y_test, test_pred)
print(f"  Confusion matrix:")
print(f"                Pred keep  Pred cancel")
print(f"  Actual keep    {cm[0,0]:6d}      {cm[0,1]:6d}")
print(f"  Actual cancel  {cm[1,0]:6d}      {cm[1,1]:6d}")


# ══════════════════════════════════════════════════════════════
# SECTION 7: PRODUCTION TALKING POINTS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 7: What I'd Do Next (Production Angle)")
print("=" * 60)
print("""
Aimé WILL ask: 'How would you deploy and monitor this?'

1. CALIBRATION
   LightGBM probabilities aren't well-calibrated by default.
   Fix: CalibratedClassifierCV(lgbm, method='sigmoid')  (Platt scaling)

2. MONITORING — track weekly:
   - PSI on feature distributions (alert if PSI > 0.25)
   - Actual cancellation rate vs model's predicted rate
   - ROC-AUC on a rolling holdout

3. RETRAINING
   - Scheduled: weekly retrain on rolling 90-day window
   - Triggered: when drift detected (PSI > 0.25 or ROC drop > 5%)

4. FAIRNESS — check ROC-AUC per segment:
   - By channel (direct vs OTA)  |  by room_type
   - 0.85 overall but 0.60 on OTA = hidden problem

5. SCALE
   - 3k rows: sklearn/lgbm fine
   - 30M rows: Spark + LightGBM distributed, or sample-train + serve sklearn
""")
print("All sections complete.")
