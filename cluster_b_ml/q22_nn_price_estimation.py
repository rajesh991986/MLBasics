"""
Q22 — Neural Network for Price Estimation [HIGH PROBABILITY — reported live task]
Target time: 20 min | Requires: torch, sklearn, numpy, pandas

APPROACH (say this in the first 60 seconds):
"I'll build a regression MLP to predict hotel prices. First I engineer features —
log-transform the target (prices are right-skewed), normalise numerics, one-hot
categoricals. Then a 3-layer NN with ReLU activations, MSE loss, Adam optimiser.
I'll track train vs val loss to spot overfitting, use dropout as regularisation,
and report MAE and RMSE on held-out test. Finally I'll discuss what I'd change
at production scale."

WHY REGRESSION NOT CLASSIFICATION:
  Price is a continuous target → use MSELoss / MAELoss, not CrossEntropyLoss
  Output layer: single neuron, no activation (raw predicted value)

KEY DECISIONS TO JUSTIFY LIVE:
  1. Log-transform target  → prices are right-skewed; log makes residuals symmetric
  2. StandardScaler        → NN gradient descent sensitive to feature scale
  3. ReLU not sigmoid      → avoids vanishing gradient in deep nets
  4. Adam not SGD          → adaptive learning rate, converges faster on tabular
  5. Dropout               → regularisation, prevents co-adaptation of neurons
  6. Batch training        → mini-batches give noisy but fast gradient estimates

BIAS-VARIANCE IN NNs:
  High bias (underfitting): increase layers/neurons, train longer
  High variance (overfitting): add dropout, L2 weight decay, more data, early stop
  Diagnostic: train loss << val loss → overfitting; both high → underfitting
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────
# SECTION 1: Synthetic hotel price dataset
# ─────────────────────────────────────────────────────────────

def make_hotel_dataset(n: int = 2000) -> pd.DataFrame:
    """
    Simulate realistic hotel pricing features.
    Price depends on: star rating, location, seasonality, lead time, room type.
    We add noise so the model can't perfectly fit — mirrors real data.
    """
    stars       = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.10, 0.35, 0.35, 0.15])
    location    = np.random.choice(["central", "suburb", "airport"], n, p=[0.4, 0.4, 0.2])
    room_type   = np.random.choice(["standard", "deluxe", "suite"], n, p=[0.5, 0.35, 0.15])
    lead_days   = np.random.randint(0, 180, n)           # days booked in advance
    stay_nights = np.random.randint(1, 14, n)
    month       = np.random.randint(1, 13, n)            # 1=Jan, 12=Dec
    is_weekend  = np.random.randint(0, 2, n)

    # Base price with realistic interactions
    base = stars * 40
    loc_premium  = np.where(location == "central", 50,
                   np.where(location == "airport", 20, 0))
    room_premium = np.where(room_type == "suite", 120,
                   np.where(room_type == "deluxe", 50, 0))
    # Seasonal: summer (Jun-Aug) and Dec are expensive
    season_mult  = np.where(np.isin(month, [6, 7, 8, 12]), 1.3, 1.0)
    # Early booking discount
    lead_discount = np.clip(lead_days * 0.2, 0, 30)
    weekend_premium = is_weekend * 15
    noise = np.random.normal(0, 20, n)

    price = (base + loc_premium + room_premium + weekend_premium
             - lead_discount) * season_mult + noise
    price = np.clip(price, 30, 800)  # realistic floor/ceiling

    return pd.DataFrame({
        "stars": stars,
        "location": location,
        "room_type": room_type,
        "lead_days": lead_days,
        "stay_nights": stay_nights,
        "month": month,
        "is_weekend": is_weekend,
        "price": price,
    })


# ─────────────────────────────────────────────────────────────
# SECTION 2: Feature engineering
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Key decisions:
    1. Log-transform target  → right-skewed distribution → more symmetric residuals
    2. Cyclical month encoding → month 12 and 1 are adjacent, sin/cos captures this
    3. One-hot categoricals  → no ordinal assumption imposed on NN
    4. StandardScaler        → zero-mean, unit-variance for stable gradient flow
    """
    df = df.copy()

    # --- Target: log-transform price ---
    # INTERVIEW: "Why log?" → prices are right-skewed (long tail of expensive hotels)
    # log makes residuals more symmetric, RMSE in log space ≈ MAPE in original
    y_raw = df["price"].values
    y_log = np.log1p(y_raw)      # log1p = log(1+x), safe for small values

    # --- Cyclical month (sin/cos encoding) ---
    # INTERVIEW: "Why not just use month as a number?" →
    # month=12 and month=1 are 11 apart numerically but only 1 step cyclically
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- One-hot encode categoricals ---
    df = pd.get_dummies(df, columns=["location", "room_type"], drop_first=False)

    # --- Drop original month (replaced by sin/cos) and target ---
    feature_cols = [c for c in df.columns if c not in ["price", "month"]]
    X = df[feature_cols].values.astype(np.float32)

    # --- Normalise features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled.astype(np.float32), y_log.astype(np.float32), y_raw, feature_cols, scaler


# ─────────────────────────────────────────────────────────────
# SECTION 3: Neural Network architecture
# ─────────────────────────────────────────────────────────────

class PriceEstimator(nn.Module):
    """
    3-layer MLP for regression.

    Architecture decisions (justify each one live):
    ┌──────────────┐
    │  Input (d)   │  ← StandardScaler'd features
    └──────┬───────┘
    ┌──────▼───────┐
    │ Linear(d→128)│
    │ BatchNorm1d  │  ← stabilises training, reduces sensitivity to lr
    │ ReLU         │  ← non-linearity; no vanishing gradient issue
    │ Dropout(0.3) │  ← randomly zero 30% of neurons → regularisation
    └──────┬───────┘
    ┌──────▼───────┐
    │ Linear(128→64│
    │ BatchNorm1d  │
    │ ReLU         │
    │ Dropout(0.2) │  ← less dropout in deeper layers (common heuristic)
    └──────┬───────┘
    ┌──────▼───────┐
    │ Linear(64→1) │  ← single output neuron, NO activation
    └──────────────┘    raw logit = predicted log(price)
    """
    def __init__(self, input_dim: int, dropout1: float = 0.3, dropout2: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout2),

            nn.Linear(64, 1),   # regression: single output, no activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # shape: (batch,)


# ─────────────────────────────────────────────────────────────
# SECTION 4: Training loop
# ─────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, epochs: int = 50, lr: float = 1e-3):
    """
    Standard PyTorch training loop.

    Loss: MSELoss on log-transformed target
    Optimiser: Adam — adaptive per-parameter lr, works well out of the box
    Early stopping: stop when val loss stops improving (prevents overfitting)

    INTERVIEW: "Why Adam over SGD?"
    → Adam adapts lr per parameter, handles sparse gradients well,
      converges faster. SGD with momentum can generalise better but
      needs careful lr scheduling. For prototyping: Adam. For production: tune.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # weight_decay=L2 regularisation — penalises large weights

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    early_stop_patience = 10

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()          # clear gradients (they accumulate!)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()                # backprop
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clip
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():             # no gradient tracking in eval
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                val_losses.append(criterion(y_pred, y_batch).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  Early stop at epoch {epoch+1} (best val={best_val_loss:.4f})")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | train={train_loss:.4f} | val={val_loss:.4f}")

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    return history


# ─────────────────────────────────────────────────────────────
# SECTION 5: Evaluation metrics
# ─────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_log_test, y_raw_test):
    """
    Metrics on original price scale (not log-transformed).

    RMSE: penalises large errors heavily (sensitive to outliers)
    MAE:  average absolute error, more robust to outliers
    MAPE: percentage error — interpretable ("predictions are off by X% on average")

    INTERVIEW: "Which metric matters here?"
    → Depends on the business:
      If a $50 error on a $100 hotel hurts as much as a $50 error on a $500 hotel
      → use MAPE (percentage error).
      If absolute dollar error matters → use MAE.
      Never report only RMSE — it's hard to interpret without context.
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        y_log_pred = model(X_t).numpy()

    # Convert back from log scale
    y_pred = np.expm1(y_log_pred)       # inverse of log1p
    y_true = y_raw_test

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae  = np.mean(np.abs(y_pred - y_true))
    mape = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8))) * 100

    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE%": round(mape, 2),
            "y_pred": y_pred, "y_true": y_true}


# ─────────────────────────────────────────────────────────────
# SECTION 6: Bias-variance diagnosis from learning curves
# ─────────────────────────────────────────────────────────────

def diagnose(history: dict) -> str:
    """
    Read the training curves to classify the failure mode.

    Pattern                          Diagnosis         Fix
    ─────────────────────────────────────────────────────────
    Both losses high                 High bias         More layers/neurons
    train << val (big gap)           High variance     Dropout, weight decay,
                                     (overfitting)     more data, early stop
    Both losses low and close        Good fit          Ship it
    Val loss increases, train drops  Overfitting       See above
    """
    final_train = history["train_loss"][-1]
    final_val   = history["val_loss"][-1]
    gap = final_val - final_train

    if final_train > 0.05 and final_val > 0.05:
        return f"HIGH BIAS (underfitting) — both losses high: train={final_train:.4f}, val={final_val:.4f}. Fix: add layers/neurons."
    elif gap > 0.02:
        return f"HIGH VARIANCE (overfitting) — gap={gap:.4f}. Fix: increase dropout, add weight decay, get more data."
    else:
        return f"GOOD FIT — train={final_train:.4f}, val={final_val:.4f}, gap={gap:.4f}."


# ─────────────────────────────────────────────────────────────
# SECTION 7: Run everything
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SECTION 1+2: Generate data + Feature Engineering")
    print("=" * 60)

    df = make_hotel_dataset(n=2000)
    print(f"  Dataset: {df.shape} | price range: ${df['price'].min():.0f}–${df['price'].max():.0f}")
    print(f"  Price distribution: mean=${df['price'].mean():.0f}, "
          f"median=${df['price'].median():.0f}, std=${df['price'].std():.0f}")

    X, y_log, y_raw, feature_cols, scaler = engineer_features(df)
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Target after log1p: mean={y_log.mean():.3f}, std={y_log.std():.3f}")

    # Train / val / test split: 70 / 15 / 15
    X_tv, X_test, y_tv, y_test_log, yraw_tv, yraw_test = train_test_split(
        X, y_log, y_raw, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val, _, _ = train_test_split(
        X_tv, y_tv, yraw_tv, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
    )
    print(f"\n  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # DataLoaders
    def to_loader(X, y, batch_size=64, shuffle=True):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train)
    val_loader   = to_loader(X_val,   y_val,   shuffle=False)

    print("\n" + "=" * 60)
    print("SECTION 3+4: Build + Train PriceEstimator")
    print("=" * 60)
    print(f"  Architecture: {len(feature_cols)} → 128 → 64 → 1")

    model = PriceEstimator(input_dim=len(feature_cols))
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train(model, train_loader, val_loader, epochs=60, lr=1e-3)

    print("\n" + "=" * 60)
    print("SECTION 5: Evaluation on held-out test set")
    print("=" * 60)
    metrics = evaluate(model, X_test, y_test_log, yraw_test)
    print(f"\n  RMSE:  ${metrics['RMSE']}")
    print(f"  MAE:   ${metrics['MAE']}")
    print(f"  MAPE:  {metrics['MAPE%']}%")

    # Sample predictions vs actuals
    print("\n  Sample predictions vs actuals:")
    print(f"  {'Predicted':>12} {'Actual':>10} {'Error':>8}")
    for pred, true in zip(metrics["y_pred"][:8], metrics["y_true"][:8]):
        print(f"  ${pred:>10.1f} ${true:>9.1f} ${abs(pred-true):>7.1f}")

    print("\n" + "=" * 60)
    print("SECTION 6: Bias-Variance Diagnosis")
    print("=" * 60)
    diagnosis = diagnose(history)
    print(f"\n  {diagnosis}")

    print("\n" + "=" * 60)
    print("KEY TALKING POINTS")
    print("=" * 60)
    print("""
Architecture decisions — justify each one:
  ReLU not sigmoid  → no vanishing gradient, faster training
  BatchNorm         → stabilises training, lets you use higher lr
  Dropout(0.3/0.2)  → regularisation, forces redundant representations
  Output: Linear(→1), no activation → regression, raw predicted value
  Loss: MSELoss on log(price) → symmetric residuals, ~MAPE in original space

Feature engineering:
  log1p(price)      → right-skewed target → symmetric after transform
  sin/cos month     → cyclic feature, month 12 and 1 are adjacent
  StandardScaler    → zero-mean, unit-variance → stable gradient descent
  One-hot cat       → no ordinal assumption imposed

Metrics:
  RMSE  → penalises large errors; hard to interpret without context
  MAE   → average absolute dollar error; robust to outliers
  MAPE  → "off by X% on average"; best for stakeholder comms

Production upgrades:
  1. Hyperparameter tuning (Optuna / grid search): hidden size, lr, dropout
  2. Embeddings for high-cardinality categoricals (hotel_id → 32-dim vector)
  3. Feature importance: SHAP values on the NN
  4. Uncertainty: predict mean + std (heteroscedastic loss / MC dropout)
  5. Monitoring: track RMSE on live data, PSI on feature distributions
""")
    print("All sections complete.")
