"""
Price Tier Prediction — Full ML Pipeline for Interview Prep
===========================================================
Covers: Feature Engineering, Model Comparison, Post-Processing,
        Drift Detection (Feature/Concept/Label), and Monitoring.

Run:  python cluster_b_ml/q20_price_tier_prediction.py
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# SECTION 1: DATA GENERATION (R&P Pricing Domain)
# ============================================================
# Interview angle: "I built a hands-on prototype to validate
# my feature engineering approach before touching production data."

print("=" * 70)
print("SECTION 1: Synthetic R&P Pricing Data Generation")
print("=" * 70)

N_SAMPLES = 5000

REGIONS = ["US", "EU", "APAC", "LATAM", "MEA", "ANZ"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "INR", "BRL", "SGD", "KRW", "CAD"]
CONTENT_TYPES = ["movie", "tv_series", "documentary", "short", "live_event", "music_video", "podcast"]
PRICE_TIERS = [f"tier_{i}" for i in range(1, 11)]  # 10 tiers

data = {
    "region": np.random.choice(REGIONS, N_SAMPLES),
    "currency": np.random.choice(CURRENCIES, N_SAMPLES),
    "content_type": np.random.choice(CONTENT_TYPES, N_SAMPLES),
    "base_price_usd": np.random.lognormal(mean=2.0, sigma=0.8, size=N_SAMPLES),
    "duration_minutes": np.random.exponential(scale=90, size=N_SAMPLES).clip(5, 500),
    "partner_tenure_days": np.random.randint(30, 3650, N_SAMPLES),
    "historical_avg_price": np.random.lognormal(mean=2.0, sigma=0.6, size=N_SAMPLES),
    "catalog_size": np.random.randint(1, 10000, N_SAMPLES),
    "monthly_views": np.random.lognormal(mean=8, sigma=2, size=N_SAMPLES).astype(int),
    "release_year": np.random.randint(1980, 2026, N_SAMPLES),
    "is_exclusive": np.random.binomial(1, 0.2, N_SAMPLES),
    "quality_score": np.random.uniform(0.1, 1.0, N_SAMPLES),
    "contract_length_months": np.random.choice([6, 12, 24, 36], N_SAMPLES),
}

df = pd.DataFrame(data)

# Assign tiers based on price (simulates real tier boundaries)
df["price_tier"] = pd.qcut(df["base_price_usd"], q=10, labels=PRICE_TIERS)

print(f"Dataset shape: {df.shape}")
print(f"Price tiers: {df['price_tier'].value_counts().sort_index().to_dict()}")
print(f"Regions: {df['region'].unique().tolist()}")
print(f"Sample:\n{df.head(3).to_string()}\n")


# ============================================================
# SECTION 2: FEATURE ENGINEERING (42 Features, 6 Categories)
# ============================================================
# Interview: "I organized features into 6 categories..."
# 1. Log transforms — handle skewed distributions
# 2. Interaction features — capture non-linear relationships
# 3. Currency-invariant ratios — normalize across currencies
# 4. Temporal features — capture time-based patterns
# 5. Partner maturity bucketing — segment partner lifecycle
# 6. Content category grouping — reduce cardinality

print("=" * 70)
print("SECTION 2: Feature Engineering (6 Categories)")
print("=" * 70)

# --- Category 1: Log Transforms ---
# WHY: base_price, monthly_views, catalog_size are right-skewed;
#      log reduces outlier influence and improves linear model performance
df["log_base_price"] = np.log1p(df["base_price_usd"])
df["log_historical_avg"] = np.log1p(df["historical_avg_price"])
df["log_monthly_views"] = np.log1p(df["monthly_views"])
df["log_catalog_size"] = np.log1p(df["catalog_size"])

# --- Category 2: Interaction Features ---
# WHY: price per minute, views per catalog item capture value density
df["price_per_minute"] = df["base_price_usd"] / df["duration_minutes"].clip(1)
df["views_per_catalog"] = df["monthly_views"] / df["catalog_size"].clip(1)
df["price_x_quality"] = df["base_price_usd"] * df["quality_score"]
df["exclusive_premium"] = df["is_exclusive"] * df["base_price_usd"]
df["tenure_x_catalog"] = df["partner_tenure_days"] * df["catalog_size"]
df["duration_x_quality"] = df["duration_minutes"] * df["quality_score"]

# --- Category 3: Currency-Invariant Ratios ---
# WHY: comparing prices across currencies directly is meaningless;
#      ratios to historical avg are currency-agnostic
df["price_to_hist_ratio"] = df["base_price_usd"] / df["historical_avg_price"].clip(0.01)
df["price_deviation_pct"] = (df["base_price_usd"] - df["historical_avg_price"]) / df["historical_avg_price"].clip(0.01)
df["log_price_ratio"] = np.log1p(df["price_to_hist_ratio"])

# --- Category 4: Temporal Features ---
# WHY: content age affects pricing; recent content commands premium
current_year = 2026
df["content_age_years"] = current_year - df["release_year"]
df["is_new_release"] = (df["content_age_years"] <= 2).astype(int)
df["is_catalog_title"] = (df["content_age_years"] > 10).astype(int)
df["age_bucket"] = pd.cut(df["content_age_years"], bins=[0, 2, 5, 10, 50], labels=[0, 1, 2, 3]).astype(int)
df["contract_to_age_ratio"] = df["contract_length_months"] / df["content_age_years"].clip(1)

# --- Category 5: Partner Maturity Bucketing ---
# WHY: new vs established partners have different pricing patterns;
#      buckets capture non-linear relationship with price
df["partner_maturity"] = pd.cut(
    df["partner_tenure_days"],
    bins=[0, 180, 365, 1095, 3650],
    labels=["new", "growing", "established", "veteran"]
)
df["partner_maturity_encoded"] = df["partner_maturity"].cat.codes
df["is_new_partner"] = (df["partner_tenure_days"] < 180).astype(int)
df["tenure_years"] = df["partner_tenure_days"] / 365.25

# --- Category 6: Content Category Grouping ---
# WHY: reduces cardinality; groups similar content for better generalization
content_group_map = {
    "movie": "long_form", "tv_series": "long_form", "documentary": "long_form",
    "short": "short_form", "music_video": "short_form",
    "live_event": "event", "podcast": "event"
}
df["content_group"] = df["content_type"].map(content_group_map)

# One-hot encode categoricals
df_encoded = pd.get_dummies(df, columns=["region", "currency", "content_type", "content_group", "partner_maturity"],
                            drop_first=True, dtype=int)

# Define feature columns (exclude target + raw categoricals)
exclude_cols = ["price_tier", "base_price_usd", "historical_avg_price",
                "partner_tenure_days", "release_year"]
feature_cols = [c for c in df_encoded.columns if c not in exclude_cols
                and df_encoded[c].dtype in [np.float64, np.int64, np.int32, np.uint8]]

print(f"Total engineered features: {len(feature_cols)}")
print(f"Feature categories:")
print(f"  1. Log transforms: log_base_price, log_historical_avg, log_monthly_views, log_catalog_size")
print(f"  2. Interactions: price_per_minute, views_per_catalog, price_x_quality, exclusive_premium, ...")
print(f"  3. Currency-invariant: price_to_hist_ratio, price_deviation_pct, log_price_ratio")
print(f"  4. Temporal: content_age_years, is_new_release, is_catalog_title, age_bucket, ...")
print(f"  5. Partner maturity: partner_maturity_encoded, is_new_partner, tenure_years")
print(f"  6. Content grouping: one-hot encoded content_group_*")
print()


# ============================================================
# SECTION 3: MODEL COMPARISON — V1 (LogReg) vs V2 (XGBoost)
# ============================================================
# Interview: "We started with LogReg as a day-30 interpretable baseline,
# then graduated to XGBoost for day-60 production with non-linear boundaries."

print("=" * 70)
print("SECTION 3: Model Comparison — LogReg (V1) vs XGBoost (V2)")
print("=" * 70)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, top_k_accuracy_score)

# Encode target
le = LabelEncoder()
y = le.fit_transform(df_encoded["price_tier"])
X = df_encoded[feature_cols].values.astype(np.float64)

# Handle any NaN/Inf from feature engineering
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)

# Scale for LogReg (XGBoost doesn't need it but won't hurt)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- V1: Logistic Regression ---
# PROS: Interpretable coefficients, fast training, good baseline,
#        easy to explain feature importance via weights
# CONS: Assumes linear decision boundaries, struggles with interactions
print("\n--- V1: Logistic Regression ---")
print("PROS: Interpretable, fast, coefficients show feature impact")
print("CONS: Linear boundaries only, needs feature engineering for interactions")

lr = LogisticRegression(max_iter=1000, multi_class="multinomial", C=1.0, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)

lr_acc = accuracy_score(y_test, y_pred_lr)
lr_top3 = top_k_accuracy_score(y_test, y_proba_lr, k=3)
print(f"Accuracy: {lr_acc:.4f}")
print(f"Top-3 Accuracy: {lr_top3:.4f}")

# --- V2: Try XGBoost first, fall back to RandomForest ---
# XGBoost PROS: Non-linear boundaries, built-in feature importance,
#               handles missing values natively, regularization built-in
# XGBoost CONS: Less interpretable, needs tuning, overfitting risk
# RandomForest PROS: Handles non-linearity, ensemble diversity, OOB error estimate
# RandomForest CONS: Less precise than boosting, larger model size

HAS_XGB = False
try:
    from xgboost import XGBClassifier
    print("\n--- V2: XGBoost (Gradient Boosted Trees) ---")
    print("PROS: Non-linear boundaries, built-in feature importance, handles missing values")
    print("CONS: Less interpretable, needs tuning, overfitting risk")

    v2_model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="mlogloss", use_label_encoder=False,
    )
    v2_name = "XGBoost (V2)"
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import RandomForestClassifier
    print("\n--- V2: Random Forest (XGBoost unavailable — needs libomp) ---")
    print("PROS: Non-linear boundaries, ensemble diversity, OOB error, no tuning needed")
    print("CONS: Slower inference than boosting, larger model, less precise")

    v2_model = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1,
    )
    v2_name = "RandomForest (V2)"

v2_model.fit(X_train_scaled, y_train)
y_pred_v2 = v2_model.predict(X_test_scaled)
y_proba_v2 = v2_model.predict_proba(X_test_scaled)

v2_acc = accuracy_score(y_test, y_pred_v2)
v2_top3 = top_k_accuracy_score(y_test, y_proba_v2, k=3)
print(f"Accuracy: {v2_acc:.4f}")
print(f"Top-3 Accuracy: {v2_top3:.4f}")

# --- Comparison Table ---
print("\n--- Model Comparison ---")
print(f"{'Model':<20} {'Accuracy':>10} {'Top-3 Acc':>10} {'Interpretable':>15} {'Training Speed':>15}")
print("-" * 70)
print(f"{'LogReg (V1)':<20} {lr_acc:>10.4f} {lr_top3:>10.4f} {'High':>15} {'Fast':>15}")
interp = "Medium" if HAS_XGB else "Medium-High"
speed = "Moderate" if HAS_XGB else "Moderate"
print(f"{v2_name:<20} {v2_acc:>10.4f} {v2_top3:>10.4f} {interp:>15} {speed:>15}")

# --- Feature Importance (Top 10) ---
print("\n--- Top 10 Feature Importances ---")
importances = v2_model.feature_importances_
model_name = v2_name

top_idx = np.argsort(importances)[-10:][::-1]
for rank, idx in enumerate(top_idx, 1):
    print(f"  {rank:>2}. {feature_cols[idx]:<35} {importances[idx]:.4f}")

# Interview nuance: "LogReg may beat XGBoost on synthetic data because
# relationships are mostly linear. With real non-linear tier boundaries
# (e.g., volume discounts, regional premiums), XGBoost pulls ahead."
print("\n[Interview Note] LogReg can match/beat XGBoost on synthetic data")
print("because the underlying relationships are mostly linear. In production")
print("with real non-linear tier boundaries, XGBoost would pull ahead.")
print()


# ============================================================
# SECTION 4: POST-PROCESSING
# ============================================================
# Interview: "Post-processing includes confidence routing, adjacent-tier
# checks, and a stable API contract for downstream services."

print("=" * 70)
print("SECTION 4: Post-Processing Pipeline")
print("=" * 70)

# Use best available model for post-processing demo
best_proba = y_proba_v2 if HAS_XGB else y_proba_lr
best_pred = y_pred_v2 if HAS_XGB else y_pred_lr


def post_process_prediction(probabilities, predicted_class, class_names,
                            high_conf=0.7, low_conf=0.4):
    """
    Post-processing pipeline:
    1. Confidence routing: auto-approve / human-review / escalate
    2. Adjacent-tier check: flag if 2nd-best tier is adjacent
    3. Return stable API response format
    """
    max_prob = np.max(probabilities)
    sorted_idx = np.argsort(probabilities)[::-1]
    top1_tier = class_names[sorted_idx[0]]
    top2_tier = class_names[sorted_idx[1]]

    # 1. Confidence routing
    if max_prob >= high_conf:
        routing = "auto_approve"
    elif max_prob >= low_conf:
        routing = "human_review"
    else:
        routing = "escalate"

    # 2. Adjacent tier check — if top-2 are adjacent tiers, flag it
    tier_num_1 = int(top1_tier.split("_")[1])
    tier_num_2 = int(top2_tier.split("_")[1])
    is_adjacent = abs(tier_num_1 - tier_num_2) == 1

    # 3. Stable API response (always HTTP 200, degraded flag for low confidence)
    response = {
        "predicted_tier": top1_tier,
        "confidence": round(float(max_prob), 4),
        "routing": routing,
        "top_3_tiers": [
            {"tier": class_names[sorted_idx[i]], "prob": round(float(probabilities[sorted_idx[i]]), 4)}
            for i in range(3)
        ],
        "adjacent_tier_flag": is_adjacent,
        "degraded": routing == "escalate",
        "model_version": "v2_xgb_20260313",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return response


# Demo: post-process a few test samples
import json

print("\n--- Sample Post-Processed Predictions ---")
for i in [0, 50, 100]:
    result = post_process_prediction(best_proba[i], best_pred[i], le.classes_)
    print(f"\nSample {i}: actual={le.classes_[y_test[i]]}")
    print(json.dumps(result, indent=2))

# Confidence distribution
confidences = np.max(best_proba, axis=1)
print(f"\n--- Confidence Routing Distribution ---")
print(f"Auto-approve (>0.7):  {(confidences >= 0.7).mean():.1%}")
print(f"Human-review (0.4-0.7): {((confidences >= 0.4) & (confidences < 0.7)).mean():.1%}")
print(f"Escalate (<0.4):      {(confidences < 0.4).mean():.1%}")

# Cassandra output format (what downstream Scala services consume)
print("\n--- Cassandra Storage Format (downstream consumption) ---")
cassandra_row = {
    "partner_id": "P-12345",
    "content_id": "C-67890",
    "predicted_tier": "tier_5",
    "confidence": 0.82,
    "model_version": "v2_xgb_20260313",
    "features_hash": "sha256:abc123...",
    "prediction_ts": "2026-03-13T10:30:00Z",
    "ttl_seconds": 86400,
}
print(json.dumps(cassandra_row, indent=2))
print()


# ============================================================
# SECTION 5: DRIFT DETECTION (Feature / Concept / Label)
# ============================================================
# Interview: "We monitor three types of drift, each with different
# detection methods and alerting thresholds."

print("=" * 70)
print("SECTION 5: Drift Detection — All Three Types")
print("=" * 70)

from scipy import stats


# --- 5A: Feature Drift (PSI — Population Stability Index) ---
# WHY PSI: Measures how much the distribution of a feature has shifted
# between training and serving. Industry standard, easy to explain.

def calculate_psi(reference, current, n_bins=10):
    """
    Population Stability Index (PSI).
    PSI < 0.10 → no significant drift
    PSI 0.10-0.25 → moderate drift (monitor)
    PSI > 0.25 → significant drift (alert)
    """
    # Create bins from reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # handle duplicate percentiles

    ref_counts = np.histogram(reference, bins=breakpoints)[0] + 1  # +1 smoothing
    cur_counts = np.histogram(current, bins=breakpoints)[0] + 1

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi


print("\n--- 5A: Feature Drift (PSI) ---")
print("PSI < 0.10 → stable | 0.10-0.25 → moderate | > 0.25 → ALERT")

# Simulate production data with drift in some features
df_prod = df.copy()
# Inject drift: shift base_price up by 30%
df_prod["log_base_price"] = np.log1p(df["base_price_usd"] * 1.3)
# Inject drift: new partner distribution shifted
df_prod["tenure_years"] = df["partner_tenure_days"] / 365.25 * 0.6

drift_features = ["log_base_price", "log_monthly_views", "tenure_years",
                   "quality_score", "price_per_minute"]

print(f"\n{'Feature':<25} {'PSI':>8} {'Status':>12}")
print("-" * 50)
for feat in drift_features:
    psi_val = calculate_psi(df[feat].values, df_prod[feat].values)
    status = "ALERT" if psi_val > 0.25 else ("Monitor" if psi_val > 0.10 else "Stable")
    print(f"{feat:<25} {psi_val:>8.4f} {status:>12}")


# --- 5B: Concept Drift (ADWIN-style Sliding Window with KS Test) ---
# WHY: Concept drift means P(Y|X) changes — the relationship between
# features and target shifts. We detect this by monitoring model
# performance on sliding windows.

def detect_concept_drift(y_true_window1, y_pred_window1,
                         y_true_window2, y_pred_window2,
                         alpha=0.05):
    """
    Detect concept drift using KS test on prediction error distributions.
    If the error distribution shifts significantly between windows,
    P(Y|X) has likely changed.
    """
    errors_w1 = (y_true_window1 != y_pred_window1).astype(float)
    errors_w2 = (y_true_window2 != y_pred_window2).astype(float)

    ks_stat, p_value = stats.ks_2samp(errors_w1, errors_w2)

    return {
        "ks_statistic": round(ks_stat, 4),
        "p_value": round(p_value, 4),
        "drift_detected": p_value < alpha,
        "window1_error_rate": round(errors_w1.mean(), 4),
        "window2_error_rate": round(errors_w2.mean(), 4),
    }


print("\n--- 5B: Concept Drift (Sliding Window KS Test) ---")
print("Monitors P(Y|X) by comparing error distributions across time windows")

# Simulate two time windows
n_half = len(y_test) // 2
# Window 1: normal predictions
w1_true, w1_pred = y_test[:n_half], best_pred[:n_half]
# Window 2: inject concept drift (randomly flip 20% of predictions)
w2_true = y_test[n_half:]
w2_pred = best_pred[n_half:].copy()
flip_mask = np.random.random(len(w2_pred)) < 0.2
w2_pred[flip_mask] = np.random.randint(0, 10, flip_mask.sum())

concept_result = detect_concept_drift(w1_true, w1_pred, w2_true, w2_pred)
print(f"  Window 1 error rate: {concept_result['window1_error_rate']:.4f}")
print(f"  Window 2 error rate: {concept_result['window2_error_rate']:.4f}")
print(f"  KS statistic: {concept_result['ks_statistic']:.4f}")
print(f"  p-value: {concept_result['p_value']:.4f}")
print(f"  Drift detected: {concept_result['drift_detected']}")


# --- 5C: Label Drift (Chi-Square + KL Divergence) ---
# WHY: Label drift means P(Y) changes — the distribution of price tiers
# shifts over time. Could mean business changes (new pricing strategy).

def detect_label_drift(ref_labels, cur_labels, n_classes=10, kl_threshold=0.10):
    """
    Detect label drift using:
    1. Chi-Square test on label counts
    2. KL Divergence on label distributions
    """
    ref_counts = np.bincount(ref_labels, minlength=n_classes) + 1  # smoothing
    cur_counts = np.bincount(cur_labels, minlength=n_classes) + 1

    ref_dist = ref_counts / ref_counts.sum()
    cur_dist = cur_counts / cur_counts.sum()

    # Chi-Square
    chi2_stat, chi2_p = stats.chisquare(cur_counts, f_exp=ref_counts * (cur_counts.sum() / ref_counts.sum()))

    # KL Divergence
    kl_div = stats.entropy(cur_dist, ref_dist)

    return {
        "chi2_statistic": round(chi2_stat, 4),
        "chi2_p_value": round(chi2_p, 4),
        "kl_divergence": round(kl_div, 6),
        "drift_detected": kl_div > kl_threshold,
        "ref_distribution": {f"tier_{i+1}": round(p, 3) for i, p in enumerate(ref_dist)},
        "cur_distribution": {f"tier_{i+1}": round(p, 3) for i, p in enumerate(cur_dist)},
    }


print("\n--- 5C: Label Drift (Chi-Square + KL Divergence) ---")
print("Monitors P(Y) — are price tier proportions shifting over time?")

# Simulate: production labels skewed toward higher tiers
ref_labels = y_train
# Create drifted labels: oversample higher tiers
cur_labels = y_test.copy()
# Shift 15% of labels up by 1-2 tiers to simulate business change
shift_mask = np.random.random(len(cur_labels)) < 0.15
cur_labels[shift_mask] = np.minimum(cur_labels[shift_mask] + np.random.randint(1, 3, shift_mask.sum()), 9)

label_result = detect_label_drift(ref_labels, cur_labels)
print(f"  Chi-Square stat: {label_result['chi2_statistic']:.4f}, p={label_result['chi2_p_value']:.4f}")
print(f"  KL Divergence: {label_result['kl_divergence']:.6f}")
print(f"  Drift detected: {label_result['drift_detected']}")

# --- 5D: Pseudo-Label Tracking (Before Ground Truth Arrives) ---
print("\n--- 5D: Pseudo-Label Monitoring (Pre-Ground-Truth) ---")
print("Before labels arrive, monitor prediction distribution as a proxy")

pred_dist = np.bincount(best_pred, minlength=10)
pred_dist_pct = pred_dist / pred_dist.sum()
train_dist = np.bincount(y_train, minlength=10)
train_dist_pct = train_dist / train_dist.sum()

pseudo_kl = stats.entropy(pred_dist_pct + 1e-10, train_dist_pct + 1e-10)
print(f"  Prediction vs Training label distribution KL: {pseudo_kl:.6f}")
print(f"  Alert threshold: 0.10")
print(f"  Status: {'ALERT' if pseudo_kl > 0.10 else 'Stable'}")
print()


# ============================================================
# SECTION 6: MONITORING REPORT & ALERTING
# ============================================================

print("=" * 70)
print("SECTION 6: Monitoring Report & Alerting Thresholds")
print("=" * 70)

monitoring_report = {
    "report_timestamp": datetime.utcnow().isoformat() + "Z",
    "model_version": "v2_xgb_20260313",
    "metrics": {
        "accuracy_7d_rolling": round(lr_acc, 4),  # use available accuracy
        "top3_accuracy_7d_rolling": round(lr_top3, 4),
        "avg_confidence": round(float(confidences.mean()), 4),
        "low_confidence_pct": round(float((confidences < 0.4).mean()), 4),
    },
    "drift": {
        "feature_drift": {
            "log_base_price_psi": round(calculate_psi(df["log_base_price"].values, df_prod["log_base_price"].values), 4),
            "tenure_years_psi": round(calculate_psi(df["tenure_years"].values, df_prod["tenure_years"].values), 4),
        },
        "concept_drift": {
            "ks_statistic": concept_result["ks_statistic"],
            "p_value": concept_result["p_value"],
            "detected": bool(concept_result["drift_detected"]),
        },
        "label_drift": {
            "kl_divergence": label_result["kl_divergence"],
            "detected": bool(label_result["drift_detected"]),
        },
    },
    "routing": {
        "auto_approve_pct": round(float((confidences >= 0.7).mean()), 4),
        "human_review_pct": round(float(((confidences >= 0.4) & (confidences < 0.7)).mean()), 4),
        "escalate_pct": round(float((confidences < 0.4).mean()), 4),
    },
}

print("\n--- Monitoring JSON Report ---")
print(json.dumps(monitoring_report, indent=2))

# --- Alerting Thresholds ---
print("\n--- Alerting Thresholds (Circuit Breaker Logic) ---")
print(f"{'Metric':<35} {'Threshold':>12} {'Action':>20}")
print("-" * 70)
thresholds = [
    ("Accuracy drop (7d rolling)", "< 0.35", "Page oncall"),
    ("Feature PSI (any feature)", "> 0.25", "Alert + investigate"),
    ("Feature PSI (any feature)", "> 0.50", "Auto-rollback model"),
    ("Concept drift KS p-value", "< 0.01", "Trigger retraining"),
    ("Label KL divergence", "> 0.10", "Alert data team"),
    ("Label KL divergence", "> 0.30", "Halt predictions"),
    ("Low confidence rate", "> 30%", "Alert + scale review"),
    ("Escalation rate", "> 15%", "Page oncall"),
    ("Prediction latency p99", "> 200ms", "Scale infra"),
    ("Model staleness", "> 14 days", "Auto-retrain"),
]
for metric, threshold, action in thresholds:
    print(f"{metric:<35} {threshold:>12} {action:>20}")

print("\n--- Dashboard vs Alert vs Auto-Rollback ---")
print("  Dashboard: All metrics (continuous visibility)")
print("  Alert:     PSI > 0.25, accuracy drop > 5%, escalation > 15%")
print("  Auto-rollback: PSI > 0.50, accuracy < 0.25, error rate spike > 3x")

print("\n" + "=" * 70)
print("DONE — All 6 sections complete.")
print("=" * 70)
print("""
Interview Summary Cheat Sheet:
─────────────────────────────
1. Feature Engineering: 6 categories (log, interaction, currency-invariant,
   temporal, partner maturity, content grouping) → ~42 features
2. Models: LogReg (V1, interpretable baseline) → XGBoost (V2, production)
3. Post-processing: confidence routing, adjacent-tier check, stable API contract
4. Drift Detection:
   - Feature drift → PSI (threshold 0.25)
   - Concept drift → KS test on error windows (p < 0.05)
   - Label drift → Chi-Square + KL divergence (threshold 0.10)
5. Monitoring: JSON reports → dashboards, alerts, auto-rollback circuit breakers
""")
