"""
Q26 — SQL Patterns for ML Interviews [REPORTED LIVE TASK]
Target time: 15 min | Requires: sqlite3, pandas

SAY THIS FIRST (30 sec):
  "I'll use sqlite3 in-memory — same SQL syntax as any prod DB.
   CASE statements, window functions, GROUP BY/HAVING, JOINs, CTEs."

REPORTED EXACT QUESTION: "one SQL using CASE statements" (Blind Nov 2025)

WINDOW FUNCTIONS — high probability at tier-1 ML interviews (~60%):
  ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, SUM OVER, NTILE
"""

import sqlite3
import pandas as pd
import numpy as np

np.random.seed(42)

# ── in-memory DB — no file needed, gets GC'd when conn closes ──
conn = sqlite3.connect(":memory:")

# bookings: 20 rows, 2 per user, varied cities/prices/cancellations
pd.DataFrame({
    "booking_id": range(1, 21),
    "user_id":  [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10],
    "city":     ["Paris","London","NYC","Tokyo","Paris","Sydney",
                 "London","NYC","Tokyo","Paris","Sydney","London",
                 "Paris","NYC","Tokyo","Sydney","London","Paris","NYC","Tokyo"],
    "price":    [120,250,180,320,95,210,340,175,290,110,
                 230,195,315,165,285,200,245,135,310,170],
    "nights":   [2,3,1,4,2,1,3,2,3,1,2,3,4,1,2,2,3,1,2,3],
    "cancelled":[0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1],
    "rating":   [4.5,3.8,4.2,None,4.8,3.5,None,4.1,4.7,4.3,
                 3.9,4.4,4.6,4.0,None,4.2,4.5,4.8,4.3,None],
}).to_sql("bookings", conn, index=False, if_exists="replace")

# users: tier/country metadata for JOIN examples
pd.DataFrame({
    "user_id":  range(1, 11),
    "tier":     ["gold","silver","silver","gold","bronze",
                 "silver","gold","bronze","silver","gold"],
    "country":  ["US","UK","US","JP","AU","UK","US","JP","UK","US"],
    "age_group":["25-34","35-44","25-34","45-54","18-24",
                 "35-44","25-34","45-54","25-34","35-44"],
}).to_sql("users", conn, index=False, if_exists="replace")


def run(label, sql):
    """Helper: print label + SQL + result table."""
    print(f"\n{'─'*60}\n  {label}\n{'─'*60}")
    print(f"  SQL:\n  {sql.strip()}")
    print(f"\n  Result:\n{pd.read_sql(sql, conn).to_string(index=False)}")


# ══════════════════════════════════════════════════════════════
# SECTION 1: CASE — the reported live question
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1: CASE Statements")
print("=" * 60)

# CASE is just an if-elif-else inside SQL — evaluated row by row
run("Classify price + quality with CASE", """
    SELECT
        booking_id, city, price,
        CASE
            WHEN price < 150 THEN 'budget'      -- if
            WHEN price < 250 THEN 'mid-range'   -- elif
            ELSE                   'premium'    -- else
        END AS price_tier,
        CASE
            WHEN cancelled = 1 THEN 'cancelled'
            WHEN rating >= 4.5  THEN 'excellent'
            WHEN rating >= 3.5  THEN 'good'
            ELSE                     'poor'
        END AS booking_quality
    FROM bookings
    ORDER BY price DESC
    LIMIT 8
""")

# CASE inside GROUP BY — bucket then count
run("Count bookings per price tier", """
    SELECT
        CASE
            WHEN price < 150 THEN 'budget'
            WHEN price < 250 THEN 'mid-range'
            ELSE                   'premium'
        END AS price_tier,
        COUNT(*)       AS num_bookings,
        AVG(price)     AS avg_price,
        SUM(cancelled) AS cancellations
    FROM bookings
    GROUP BY price_tier   -- group on the CASE expression
    ORDER BY avg_price
""")


# ══════════════════════════════════════════════════════════════
# SECTION 2: WINDOW FUNCTIONS — highest probability
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: Window Functions")
print("=" * 60)

# OVER (PARTITION BY ... ORDER BY ...) — runs across a "window" of rows
# unlike GROUP BY, it does NOT collapse rows

run("ROW_NUMBER — rank by price within each city", """
    SELECT
        booking_id, city, price,
        ROW_NUMBER() OVER (
            PARTITION BY city        -- restart counter for each city
            ORDER BY price DESC      -- highest price = rank 1
        ) AS rank_in_city
    FROM bookings
    ORDER BY city, rank_in_city
""")

# cumulative SUM — running total, keeps individual rows
run("Cumulative revenue per user", """
    SELECT
        user_id, booking_id,
        price * nights AS revenue,
        SUM(price * nights) OVER (
            PARTITION BY user_id     -- separate running total per user
            ORDER BY booking_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_revenue
    FROM bookings
    ORDER BY user_id, booking_id
""")

# LAG — look back 1 row within the same partition
run("LAG — price delta between bookings per user", """
    SELECT
        user_id, booking_id, price,
        LAG(price, 1) OVER (PARTITION BY user_id ORDER BY booking_id) AS prev_price,
        price - LAG(price, 1) OVER (PARTITION BY user_id ORDER BY booking_id) AS delta
    FROM bookings
    ORDER BY user_id, booking_id
""")

# NTILE(n) — divide rows into n equal buckets (like pd.qcut)
run("NTILE(4) — revenue quartiles", """
    SELECT
        booking_id, city,
        price * nights AS revenue,
        NTILE(4) OVER (ORDER BY price * nights) AS quartile  -- 1=bottom, 4=top
    FROM bookings
    ORDER BY quartile, revenue
""")


# ══════════════════════════════════════════════════════════════
# SECTION 3: GROUP BY + HAVING + JOIN
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3: GROUP BY / HAVING / JOIN")
print("=" * 60)

# HAVING filters AFTER aggregation; WHERE filters BEFORE
run("Cities: avg revenue > 300 AND cancel rate < 30%", """
    SELECT
        city,
        COUNT(*)                      AS bookings,
        ROUND(AVG(price * nights), 2) AS avg_revenue,
        ROUND(AVG(cancelled), 3)      AS cancel_rate
    FROM bookings
    GROUP BY city
    HAVING avg_revenue > 300        -- post-aggregation filter (HAVING not WHERE)
       AND cancel_rate < 0.3
    ORDER BY avg_revenue DESC
""")

# INNER JOIN — only rows where user_id exists in both tables
run("Revenue by user tier (JOIN)", """
    SELECT
        u.tier,
        COUNT(b.booking_id)              AS bookings,
        ROUND(SUM(b.price * b.nights),2) AS total_revenue,
        ROUND(AVG(b.rating), 2)          AS avg_rating
    FROM bookings b
    JOIN users u ON b.user_id = u.user_id   -- INNER JOIN (default)
    GROUP BY u.tier
    ORDER BY total_revenue DESC
""")


# ══════════════════════════════════════════════════════════════
# SECTION 4: SUBQUERIES + CTEs
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 4: Subqueries and CTEs")
print("=" * 60)

# subquery in FROM — compute city avg, then join back to find above-avg rows
run("Bookings priced above their city average (subquery)", """
    SELECT b.booking_id, b.city, b.price,
           city_avg.avg_price,
           ROUND(b.price - city_avg.avg_price, 2) AS above_by
    FROM bookings b
    JOIN (
        SELECT city, AVG(price) AS avg_price   -- subquery: one avg per city
        FROM bookings
        GROUP BY city
    ) city_avg ON b.city = city_avg.city
    WHERE b.price > city_avg.avg_price
    ORDER BY above_by DESC
    LIMIT 6
""")

# WITH (CTE) — named temp result, more readable than nested subqueries
run("CTE — top-revenue booking per user", """
    WITH ranked AS (                              -- define temp table 'ranked'
        SELECT
            user_id, booking_id, city,
            price * nights AS revenue,
            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY price * nights DESC
            ) AS rn                              -- rn=1 is the best booking
        FROM bookings
    )
    SELECT user_id, booking_id, city, revenue
    FROM ranked
    WHERE rn = 1                                 -- keep only the best
    ORDER BY revenue DESC
""")

# chained CTEs — second CTE references the first
run("Chained CTEs — users spending above average", """
    WITH user_spend AS (                          -- CTE 1: total spend per user
        SELECT user_id, SUM(price * nights) AS total_spend
        FROM bookings
        WHERE cancelled = 0
        GROUP BY user_id
    ),
    avg_spend AS (                                -- CTE 2: average across users
        SELECT AVG(total_spend) AS avg_total
        FROM user_spend
    )
    SELECT
        us.user_id, u.tier,
        ROUND(us.total_spend, 2)                        AS total_spend,
        ROUND(av.avg_total, 2)                          AS avg_spend,
        ROUND(us.total_spend / av.avg_total, 2)         AS spend_ratio
    FROM user_spend us
    JOIN users u ON us.user_id = u.user_id
    CROSS JOIN avg_spend av                      -- CROSS JOIN: single-row table, safe here
    WHERE us.total_spend > av.avg_total
    ORDER BY total_spend DESC
""")


# ══════════════════════════════════════════════════════════════
# SECTION 5: NULL HANDLING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 5: NULL Handling")
print("=" * 60)

# COALESCE returns the first non-null value in the list
run("COALESCE — impute missing ratings with city average", """
    SELECT
        b.booking_id, b.city,
        b.rating                                    AS original_rating,
        ROUND(AVG(b2.rating), 2)                    AS city_avg,
        COALESCE(b.rating, ROUND(AVG(b2.rating),2)) AS imputed_rating  -- fallback to avg
    FROM bookings b
    JOIN bookings b2 ON b.city = b2.city AND b2.rating IS NOT NULL
    GROUP BY b.booking_id, b.city, b.rating
    HAVING b.rating IS NULL
    LIMIT 5
""")


# ══════════════════════════════════════════════════════════════
# SECTION 6: SQL ↔ PANDAS CHEAT SHEET
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 6: SQL ↔ Pandas Equivalents")
print("=" * 60)
print("""
  SQL                                      Pandas
  ──────────────────────────────────────────────────────────────────
  SELECT col FROM t                        df[['col']]
  WHERE col > 5                            df[df['col'] > 5]
  GROUP BY city                            df.groupby('city')
  HAVING COUNT(*) > 2                      .filter(lambda x: len(x) > 2)
  ORDER BY rev DESC                        .sort_values('rev', ascending=False)
  LIMIT 10                                 .head(10)
  INNER JOIN t2 ON key                     df.merge(t2, on='key', how='inner')
  LEFT JOIN t2 ON key                      df.merge(t2, on='key', how='left')
  CASE WHEN x>5 THEN 'high' ELSE 'low'    np.where(df['x']>5, 'high', 'low')
  ROW_NUMBER() OVER (PARTITION BY city)    df.groupby('city').cumcount()
  LAG(col,1) OVER (ORDER BY date)          df['col'].shift(1)
  SUM() OVER (PARTITION BY user_id)        df.groupby('user_id')['col'].cumsum()
  NTILE(4) OVER (ORDER BY col)             pd.qcut(df['col'], q=4)
  COALESCE(col, default)                   df['col'].fillna(default)
  COUNT(DISTINCT user_id)                  df['user_id'].nunique()
""")

conn.close()
print("All sections complete.")
