"""
Q25 — Pandas Data Wrangling [ALMOST CERTAIN WARM-UP]
Target time: 15 min | Requires: pandas, numpy

SAY THIS FIRST (30 sec):
  "Load → inspect shape/dtypes/nulls → fix invalids → impute →
   standardise → engineer features → aggregate → encode for model."

MUSCLE MEMORY — type without thinking:
  df.shape  df.dtypes  df.isnull().sum()  df.describe()
  df['col'].fillna(df['col'].median())
  df.groupby('col').agg({'x': ['mean','sum','count']})
  df[df['a'] > 0]  df.merge(other, on='key', how='left')
  pd.get_dummies(df, columns=['cat'], drop_first=False)
"""

import numpy as np
import pandas as pd
import re
import string

np.random.seed(42)

# ── simulates what the interviewer hands you as a CSV ──────────
def make_messy_bookings(n=200):
    cities     = ["Paris", "London", "NYC", "Tokyo", "Sydney"]
    room_types = ["Standard", "Deluxe", "Suite", None]          # real nulls
    sources    = ["mobile", "DESKTOP", "Mobile", "desktop", "tablet"]  # messy case

    df = pd.DataFrame({
        "booking_id": range(1001, 1001 + n),
        "city":       np.random.choice(cities, n),
        "room_type":  np.random.choice(room_types, n),          # ~25% null
        "source":     np.random.choice(sources, n),             # mixed case bug
        "price":      np.random.normal(200, 80, n).round(2),
        "nights":     np.random.randint(1, 14, n),
        "rating":     np.where(np.random.rand(n) < 0.15, np.nan,
                               np.random.uniform(1, 5, n).round(1)),  # 15% null
        "review_text": [
            np.random.choice([
                "Great location!  Loved it.",
                "Terrible service... staff was RUDE!!!",
                "Clean rooms & excellent breakfast :)",
                "Perfect spot -- walking distance to attractions",
                None,
            ]) for _ in range(n)
        ],
        "cancelled": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    })
    df.loc[5,  "price"]  = -50    # invalid negative
    df.loc[10, "price"]  = 9999   # outlier
    df.loc[15, "nights"] = 0      # invalid zero
    return df


# ══════════════════════════════════════════════════════════════
# SECTION 1: INSPECT — first 5 things you always do
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1: Load and Inspect")
print("=" * 60)

df = make_messy_bookings()

print(f"\nShape:   {df.shape}")                                 # rows × cols
print(f"\nDtypes:\n{df.dtypes}")                               # catch object cols that should be numeric
print(f"\nNull counts:\n{df.isnull().sum()}")                  # where to impute
print(f"\nSample:\n{df.head(3).to_string()}")                  # eyeball the data
print(f"\nNumeric summary:\n{df[['price','nights','rating']].describe().round(2)}")  # spot outliers via min/max


# ══════════════════════════════════════════════════════════════
# SECTION 2: CLEAN — the live coding core
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: Clean")
print("=" * 60)

df_clean = df.copy()  # never mutate the original

# RULE: fix invalids BEFORE imputing — otherwise you impute bad values
df_clean = df_clean[df_clean["price"]  > 0]      # drop negative prices
df_clean = df_clean[df_clean["nights"] > 0]      # drop zero-night stays
df_clean = df_clean[df_clean["price"]  < 2000]   # drop obvious outliers

# numeric nulls → median (not mean — median is robust to the outliers we just missed)
df_clean["rating"] = df_clean["rating"].fillna(df_clean["rating"].median())

# categorical nulls → mode (most common value)
df_clean["room_type"] = df_clean["room_type"].fillna(df_clean["room_type"].mode()[0])

# text nulls → empty string (can't impute meaning)
df_clean["review_text"] = df_clean["review_text"].fillna("")

# standardise case — "DESKTOP" and "desktop" are the same thing
df_clean["source"] = df_clean["source"].str.lower().str.strip()

# derived features — price × nights is a useful signal
df_clean["total_revenue"]  = df_clean["price"] * df_clean["nights"]
df_clean["price_per_night"] = df_clean["price"]

print(f"\nRows after cleaning: {len(df_clean)} (removed {len(df) - len(df_clean)} bad rows)")
print(f"Nulls remaining:\n{df_clean.isnull().sum()}")           # should all be 0
print(f"\nStandardised sources: {sorted(df_clean['source'].unique())}")


# ══════════════════════════════════════════════════════════════
# SECTION 3: TEXT CLEANING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3: Text Cleaning")
print("=" * 60)

def clean_text(text: str) -> str:
    """lowercase → strip punctuation → collapse spaces → strip edges"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # "RUDE!!!" → "rude   "
    text = re.sub(r"\s+", " ", text)                                  # collapse multiple spaces
    return text.strip()

df_clean["review_clean"] = df_clean["review_text"].apply(clean_text)

# print before/after to show it works
samples = df_clean[df_clean["review_text"] != ""][["review_text","review_clean"]].head(4)
for _, row in samples.iterrows():
    print(f"  BEFORE: {row['review_text']}")
    print(f"  AFTER:  {row['review_clean']}\n")


# ══════════════════════════════════════════════════════════════
# SECTION 4: GROUPBY AGGREGATION
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 4: GroupBy Aggregations")
print("=" * 60)

# named aggregation syntax — readable, avoids MultiIndex column names
city_stats = (
    df_clean
    .groupby("city")
    .agg(
        total_bookings    = ("booking_id",    "count"),
        avg_price         = ("price",         "mean"),
        total_revenue     = ("total_revenue", "sum"),
        avg_rating        = ("rating",        "mean"),
        cancellation_rate = ("cancelled",     "mean"),
    )
    .round(2)
    .sort_values("total_revenue", ascending=False)  # rank by revenue
)
print(f"\nCity performance:\n{city_stats.to_string()}")

# pivot: rows=room_type, cols=source → mean revenue per cell
pivot = (
    df_clean
    .groupby(["room_type", "source"])["total_revenue"]
    .mean()
    .round(2)
    .unstack(fill_value=0)   # source becomes columns; 0 where no data
)
print(f"\nAvg revenue by room × source:\n{pivot.to_string()}")


# ══════════════════════════════════════════════════════════════
# SECTION 5: FILTER / SORT / RANK
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 5: Filter, Sort, Rank")
print("=" * 60)

# top 5 by revenue — standard sort + head
top5 = (
    df_clean
    .sort_values("total_revenue", ascending=False)
    [["booking_id","city","room_type","price","nights","total_revenue"]]
    .head(5)
)
print(f"\nTop 5 revenue bookings:\n{top5.to_string(index=False)}")

# compound filter — & not 'and', wrap each condition in ()
high_quality = df_clean[(df_clean["rating"] >= 4.0) & (df_clean["cancelled"] == 0)]
print(f"\nHigh-quality (rating≥4, not cancelled): {len(high_quality)}")

# within-group percentile rank — rank(pct=True) gives 0→1 percentile
df_clean["price_rank_in_city"] = (
    df_clean.groupby("city")["price"]
    .rank(pct=True)
    .round(2)
)
print(f"\nPrice percentile rank sample:\n"
      f"{df_clean[['booking_id','city','price','price_rank_in_city']].head(6).to_string(index=False)}")


# ══════════════════════════════════════════════════════════════
# SECTION 6: MERGE / JOIN
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 6: Merge")
print("=" * 60)

# second table — simulates a lookup/dimension table
city_meta = pd.DataFrame({
    "city":    ["Paris","London","NYC","Tokyo","Sydney"],
    "country": ["France","UK","USA","Japan","Australia"],
    "avg_tourist_spend_usd": [180, 200, 220, 160, 175],
})

# left join — keeps all rows from df_clean, adds matching cols from city_meta
df_merged = df_clean.merge(city_meta, on="city", how="left")
print(f"\nAfter left merge: {df_merged.shape}")
print(f"Nulls in country: {df_merged['country'].isnull().sum()} (0 = all rows matched)")
print(f"\nSample:\n{df_merged[['booking_id','city','country','price','avg_tourist_spend_usd']].head(4).to_string(index=False)}")


# ══════════════════════════════════════════════════════════════
# SECTION 7: ONE-HOT ENCODE → MODEL-READY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 7: Encode for Modelling")
print("=" * 60)

model_df = df_merged[["price","nights","rating","cancelled",
                       "city","room_type","source"]].copy()

before_cols = model_df.shape[1]
# drop_first=False → keep all dummies (tree models need all; linear use drop_first=True)
model_df = pd.get_dummies(model_df, columns=["city","room_type","source"], drop_first=False)
after_cols = model_df.shape[1]

print(f"\nCols before: {before_cols}  →  after: {after_cols}")
print(f"Example new cols: {[c for c in model_df.columns if '_' in c][:8]}...")
print(f"All numeric, zero nulls: {model_df.isnull().sum().sum() == 0}")  # must be True

print("""
KEY POINTS TO SAY:
  drop_first=False   → better for tree models; prevents dummy trap for linear
  StandardScaler     → fit on TRAIN only, transform train+val+test (prevents leakage)
  OrdinalEncoder     → use instead of get_dummies for high-cardinality (>20 values)
""")
print("All sections complete.")
