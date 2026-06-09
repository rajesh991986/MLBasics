"""
Q25 — Pandas Data Wrangling [ALMOST CERTAIN WARM-UP]
Target time: 15 min | Requires: pandas, numpy

APPROACH (say this first 30 seconds):
"I'll load the data, check shape/dtypes/nulls, then clean and transform.
For nulls: numeric → fill with mean/median; categorical → fill with mode
or 'Unknown'. For text: lowercase, strip punctuation, trim whitespace.
Then aggregate with groupby. I'll talk through every decision."

MUSCLE MEMORY — have these ready to type without thinking:
  df.shape, df.dtypes, df.isnull().sum()
  df['col'].fillna(df['col'].mean())
  df.groupby('col').agg({'metric': ['mean','sum','count']})
  df[df['col'] > threshold]
  df.merge(other, on='key', how='left')
  df['col'].apply(lambda x: ...)
  df.sort_values('col', ascending=False).head(10)
  pd.get_dummies(df, columns=['cat_col'], drop_first=True)
"""

import numpy as np
import pandas as pd
import re
import string

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# SECTION 1: Create messy hotel bookings dataset
# (interviewer will give you a CSV — this simulates it)
# ─────────────────────────────────────────────────────────────

def make_messy_bookings(n=200):
    cities     = ["Paris", "London", "NYC", "Tokyo", "Sydney"]
    room_types = ["Standard", "Deluxe", "Suite", None]   # some nulls
    sources    = ["mobile", "DESKTOP", "Mobile", "desktop", "tablet"]  # messy case

    df = pd.DataFrame({
        "booking_id":   range(1001, 1001 + n),
        "city":         np.random.choice(cities, n),
        "room_type":    np.random.choice(room_types, n),        # has nulls
        "source":       np.random.choice(sources, n),           # inconsistent case
        "price":        np.random.normal(200, 80, n).round(2),
        "nights":       np.random.randint(1, 14, n),
        "rating":       np.where(np.random.rand(n) < 0.15, np.nan,
                                 np.random.uniform(1, 5, n).round(1)),  # 15% null
        "review_text":  [
            np.random.choice([
                "Great location!  Loved it.",
                "Terrible service... staff was RUDE!!!",
                "Clean rooms & excellent breakfast :)",
                "Perfect spot -- walking distance to attractions",
                None,
            ])
            for _ in range(n)
        ],
        "cancelled":    np.random.choice([0, 1], n, p=[0.85, 0.15]),
    })
    # Inject some outliers and negatives
    df.loc[5,  "price"] = -50     # invalid
    df.loc[10, "price"] = 9999    # outlier
    df.loc[15, "nights"] = 0      # invalid
    return df


print("=" * 60)
print("SECTION 1: Load and Inspect")
print("=" * 60)

df = make_messy_bookings()

# ── First 5 things you ALWAYS do ──
print(f"\nShape:   {df.shape}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nNull counts:\n{df.isnull().sum()}")
print(f"\nSample:\n{df.head(3).to_string()}")
print(f"\nNumeric summary:\n{df[['price','nights','rating']].describe().round(2)}")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Clean — the live coding core
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2: Clean")
print("=" * 60)

df_clean = df.copy()

# 1. Fix invalid values BEFORE imputing
df_clean = df_clean[df_clean["price"] > 0]     # drop negative prices
df_clean = df_clean[df_clean["nights"] > 0]    # drop zero-night stays
df_clean = df_clean[df_clean["price"] < 2000]  # cap obvious outliers

# 2. Impute nulls
#    numeric → median (robust to outliers)
df_clean["rating"] = df_clean["rating"].fillna(df_clean["rating"].median())
#    categorical → mode
df_clean["room_type"] = df_clean["room_type"].fillna(
    df_clean["room_type"].mode()[0]
)
#    text → empty string (can't impute meaning)
df_clean["review_text"] = df_clean["review_text"].fillna("")

# 3. Standardise categorical case
df_clean["source"] = df_clean["source"].str.lower().str.strip()

# 4. Derived features
df_clean["total_revenue"] = df_clean["price"] * df_clean["nights"]
df_clean["price_per_night"] = df_clean["price"]   # already per night here

print(f"\nRows after cleaning: {len(df_clean)} (removed {len(df) - len(df_clean)} bad rows)")
print(f"Nulls remaining:\n{df_clean.isnull().sum()}")
print(f"\nStandardised sources: {sorted(df_clean['source'].unique())}")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Text cleaning
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: Text Cleaning")
print("=" * 60)

def clean_text(text: str) -> str:
    """
    Standard text cleaning pipeline:
    1. Lowercase
    2. Remove punctuation / special characters
    3. Collapse whitespace
    4. Strip leading/trailing spaces
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df_clean["review_clean"] = df_clean["review_text"].apply(clean_text)

# Show before/after
samples = df_clean[df_clean["review_text"] != ""][["review_text","review_clean"]].head(4)
for _, row in samples.iterrows():
    print(f"  BEFORE: {row['review_text']}")
    print(f"  AFTER:  {row['review_clean']}")
    print()


# ─────────────────────────────────────────────────────────────
# SECTION 4: GroupBy aggregations
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 4: GroupBy Aggregations")
print("=" * 60)

# Revenue and rating by city
city_stats = (
    df_clean
    .groupby("city")
    .agg(
        total_bookings = ("booking_id",      "count"),
        avg_price      = ("price",           "mean"),
        total_revenue  = ("total_revenue",   "sum"),
        avg_rating     = ("rating",          "mean"),
        cancellation_rate = ("cancelled",    "mean"),
    )
    .round(2)
    .sort_values("total_revenue", ascending=False)
)
print(f"\nCity performance:\n{city_stats.to_string()}")

# Revenue by room type and source
pivot = (
    df_clean
    .groupby(["room_type", "source"])["total_revenue"]
    .mean()
    .round(2)
    .unstack(fill_value=0)
)
print(f"\nAvg revenue by room × source:\n{pivot.to_string()}")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Filter + sort + rank (common live tasks)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5: Filter, Sort, Rank")
print("=" * 60)

# Top 5 highest-revenue bookings
top5 = (
    df_clean
    .sort_values("total_revenue", ascending=False)
    [["booking_id","city","room_type","price","nights","total_revenue"]]
    .head(5)
)
print(f"\nTop 5 revenue bookings:\n{top5.to_string(index=False)}")

# Bookings with rating > 4 AND not cancelled
high_quality = df_clean[(df_clean["rating"] >= 4.0) & (df_clean["cancelled"] == 0)]
print(f"\nHigh-quality bookings (rating≥4, not cancelled): {len(high_quality)}")

# Percentile ranking of price within each city
df_clean["price_rank_in_city"] = (
    df_clean.groupby("city")["price"]
    .rank(pct=True)
    .round(2)
)
print(f"\nPrice percentile rank sample:\n"
      f"{df_clean[['booking_id','city','price','price_rank_in_city']].head(6).to_string(index=False)}")


# ─────────────────────────────────────────────────────────────
# SECTION 6: Merge / join
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 6: Merge")
print("=" * 60)

# Simulate a second table: city metadata
city_meta = pd.DataFrame({
    "city":       ["Paris", "London", "NYC", "Tokyo", "Sydney"],
    "country":    ["France","UK","USA","Japan","Australia"],
    "avg_tourist_spend_usd": [180, 200, 220, 160, 175],
})

df_merged = df_clean.merge(city_meta, on="city", how="left")
print(f"\nAfter left merge with city metadata: {df_merged.shape}")
print(f"Nulls in country: {df_merged['country'].isnull().sum()} (0 = good merge)")
print(f"\nSample:\n{df_merged[['booking_id','city','country','price','avg_tourist_spend_usd']].head(4).to_string(index=False)}")


# ─────────────────────────────────────────────────────────────
# SECTION 7: One-hot encoding (prep for model)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 7: Encode for Modelling")
print("=" * 60)

model_df = df_merged[["price","nights","rating","cancelled",
                       "city","room_type","source"]].copy()

before_cols = model_df.shape[1]
model_df = pd.get_dummies(model_df, columns=["city","room_type","source"],
                          drop_first=False)  # keep all dummies, explain why
after_cols = model_df.shape[1]

print(f"\nColumns before encoding: {before_cols}")
print(f"Columns after encoding:  {after_cols}")
print(f"New columns: {[c for c in model_df.columns if '_' in c][:8]}...")
print(f"\nReady for sklearn — all numeric, no nulls: {model_df.isnull().sum().sum() == 0}")

print("""
INTERVIEW TALKING POINTS:
  drop_first=False   → keep all dummies (better for tree models; for linear use True)
  pd.get_dummies     → fast for low-cardinality; use OrdinalEncoder for high-cardinality
  StandardScaler     → apply AFTER split (fit on train, transform both)
  Why after split?   → prevent data leakage (test mean leaking into train scaling)
""")
print("All sections complete.")
