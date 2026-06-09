"""
Q26 — SQL Patterns for ML Interviews [REPORTED LIVE TASK]
Target time: 15 min | Requires: pandas (simulates SQL via sqlite3)

APPROACH (say first 30 seconds):
"I'll use sqlite3 to run real SQL on an in-memory DB — same syntax as
any production DB. I'll cover: CASE statements, window functions,
GROUP BY / HAVING, JOINs, subqueries, and CTEs. These are the patterns
that come up in every ML interview with a data wrangling component."

REPORTED EXACT QUESTION (Blind Nov 2025):
  "one SQL using case statements" — almost certain warm-up

WINDOW FUNCTIONS ARE HIGH PROBABILITY:
  ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, SUM OVER, AVG OVER
  These appear in ~60% of ML data rounds at tier-1 companies.
"""

import sqlite3
import pandas as pd
import numpy as np

np.random.seed(42)


# ─────────────────────────────────────────────────────────────
# Setup: create in-memory SQLite DB with realistic tables
# ─────────────────────────────────────────────────────────────

conn = sqlite3.connect(":memory:")

# bookings table
pd.DataFrame({
    "booking_id": range(1, 21),
    "user_id":    [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10],
    "city":       ["Paris","London","NYC","Tokyo","Paris","Sydney",
                   "London","NYC","Tokyo","Paris","Sydney","London",
                   "Paris","NYC","Tokyo","Sydney","London","Paris",
                   "NYC","Tokyo"],
    "price":      [120,250,180,320,95,210,340,175,290,110,
                   230,195,315,165,285,200,245,135,310,170],
    "nights":     [2,3,1,4,2,1,3,2,3,1,2,3,4,1,2,2,3,1,2,3],
    "cancelled":  [0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1],
    "rating":     [4.5,3.8,4.2,None,4.8,3.5,None,4.1,4.7,4.3,
                   3.9,4.4,4.6,4.0,None,4.2,4.5,4.8,4.3,None],
}).to_sql("bookings", conn, index=False, if_exists="replace")

# users table
pd.DataFrame({
    "user_id":   range(1, 11),
    "tier":      ["gold","silver","silver","gold","bronze",
                  "silver","gold","bronze","silver","gold"],
    "country":   ["US","UK","US","JP","AU","UK","US","JP","UK","US"],
    "age_group": ["25-34","35-44","25-34","45-54","18-24",
                  "35-44","25-34","45-54","25-34","35-44"],
}).to_sql("users", conn, index=False, if_exists="replace")


def run(label, sql):
    """Run SQL query and print result."""
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  SQL:\n  {sql.strip()}")
    result = pd.read_sql(sql, conn)
    print(f"\n  Result:\n{result.to_string(index=False)}")
    return result


# ─────────────────────────────────────────────────────────────
# SECTION 1: CASE statements — the reported live question
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 1: CASE Statements")
print("=" * 60)

run("Price tier classification with CASE", """
    SELECT
        booking_id,
        city,
        price,
        CASE
            WHEN price < 150 THEN 'budget'
            WHEN price < 250 THEN 'mid-range'
            ELSE                   'premium'
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

run("Count by price tier (CASE inside GROUP BY)", """
    SELECT
        CASE
            WHEN price < 150 THEN 'budget'
            WHEN price < 250 THEN 'mid-range'
            ELSE                   'premium'
        END AS price_tier,
        COUNT(*)         AS num_bookings,
        AVG(price)       AS avg_price,
        SUM(cancelled)   AS cancellations
    FROM bookings
    GROUP BY price_tier
    ORDER BY avg_price
""")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Window functions — highest probability
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2: Window Functions")
print("=" * 60)

run("ROW_NUMBER — rank bookings by price within each city", """
    SELECT
        booking_id,
        city,
        price,
        ROW_NUMBER() OVER (PARTITION BY city ORDER BY price DESC) AS rank_in_city
    FROM bookings
    ORDER BY city, rank_in_city
""")

run("Running total revenue per user (cumulative SUM)", """
    SELECT
        user_id,
        booking_id,
        price * nights AS revenue,
        SUM(price * nights) OVER (
            PARTITION BY user_id
            ORDER BY booking_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_revenue
    FROM bookings
    ORDER BY user_id, booking_id
""")

run("LAG — price change between consecutive bookings per user", """
    SELECT
        user_id,
        booking_id,
        price,
        LAG(price, 1) OVER (PARTITION BY user_id ORDER BY booking_id) AS prev_price,
        price - LAG(price, 1) OVER (PARTITION BY user_id ORDER BY booking_id) AS price_delta
    FROM bookings
    ORDER BY user_id, booking_id
""")

run("NTILE — split bookings into 4 revenue quartiles", """
    SELECT
        booking_id,
        city,
        price * nights AS revenue,
        NTILE(4) OVER (ORDER BY price * nights) AS revenue_quartile
    FROM bookings
    ORDER BY revenue_quartile, revenue
""")


# ─────────────────────────────────────────────────────────────
# SECTION 3: GROUP BY + HAVING + JOIN
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: GROUP BY / HAVING / JOIN")
print("=" * 60)

run("Cities with avg revenue > 500 AND < 20% cancellation (HAVING)", """
    SELECT
        city,
        COUNT(*)                        AS bookings,
        ROUND(AVG(price * nights), 2)   AS avg_revenue,
        ROUND(AVG(cancelled), 3)        AS cancel_rate
    FROM bookings
    GROUP BY city
    HAVING avg_revenue > 300
       AND cancel_rate < 0.3
    ORDER BY avg_revenue DESC
""")

run("JOIN bookings with users — revenue by tier", """
    SELECT
        u.tier,
        COUNT(b.booking_id)              AS bookings,
        ROUND(SUM(b.price * b.nights),2) AS total_revenue,
        ROUND(AVG(b.rating), 2)          AS avg_rating,
        ROUND(AVG(b.cancelled), 3)       AS cancel_rate
    FROM bookings b
    JOIN users u ON b.user_id = u.user_id
    GROUP BY u.tier
    ORDER BY total_revenue DESC
""")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Subqueries + CTEs
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 4: Subqueries and CTEs")
print("=" * 60)

run("Subquery — bookings above the city average price", """
    SELECT b.booking_id, b.city, b.price,
           city_avg.avg_price,
           ROUND(b.price - city_avg.avg_price, 2) AS above_avg_by
    FROM bookings b
    JOIN (
        SELECT city, AVG(price) AS avg_price
        FROM bookings
        GROUP BY city
    ) city_avg ON b.city = city_avg.city
    WHERE b.price > city_avg.avg_price
    ORDER BY above_avg_by DESC
    LIMIT 6
""")

run("CTE — top booking per user by revenue", """
    WITH ranked AS (
        SELECT
            user_id,
            booking_id,
            city,
            price * nights AS revenue,
            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY price * nights DESC
            ) AS rn
        FROM bookings
    )
    SELECT user_id, booking_id, city, revenue
    FROM ranked
    WHERE rn = 1
    ORDER BY revenue DESC
""")

run("CTE chain — users whose total spend > avg user spend", """
    WITH user_spend AS (
        SELECT user_id, SUM(price * nights) AS total_spend
        FROM bookings
        WHERE cancelled = 0
        GROUP BY user_id
    ),
    avg_spend AS (
        SELECT AVG(total_spend) AS avg_total
        FROM user_spend
    )
    SELECT
        us.user_id,
        u.tier,
        ROUND(us.total_spend, 2)       AS total_spend,
        ROUND(av.avg_total, 2)         AS avg_spend,
        ROUND(us.total_spend / av.avg_total, 2) AS spend_ratio
    FROM user_spend us
    JOIN users u ON us.user_id = u.user_id
    CROSS JOIN avg_spend av
    WHERE us.total_spend > av.avg_total
    ORDER BY total_spend DESC
""")


# ─────────────────────────────────────────────────────────────
# SECTION 5: NULL handling + COALESCE + NULLIF
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5: NULL Handling")
print("=" * 60)

run("COALESCE — fill missing ratings with city average", """
    SELECT
        b.booking_id,
        b.city,
        b.rating                                     AS original_rating,
        ROUND(AVG(b2.rating), 2)                     AS city_avg_rating,
        COALESCE(b.rating, ROUND(AVG(b2.rating),2))  AS imputed_rating
    FROM bookings b
    JOIN bookings b2 ON b.city = b2.city AND b2.rating IS NOT NULL
    GROUP BY b.booking_id, b.city, b.rating
    HAVING b.rating IS NULL
    LIMIT 5
""")


# ─────────────────────────────────────────────────────────────
# SECTION 6: Pandas equivalents side by side
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 6: SQL ↔ Pandas Equivalents")
print("=" * 60)
print("""
  SQL                                    Pandas
  ────────────────────────────────────────────────────────────────────
  SELECT col FROM t                      df[['col']]
  WHERE col > 5                          df[df['col'] > 5]
  GROUP BY city                          df.groupby('city')
  HAVING COUNT(*) > 2                    .filter(lambda x: len(x) > 2)
  ORDER BY revenue DESC                  .sort_values('revenue', ascending=False)
  LIMIT 10                               .head(10)
  JOIN t2 ON key                         df.merge(t2, on='key', how='inner')
  LEFT JOIN                              df.merge(t2, on='key', how='left')
  CASE WHEN x>5 THEN 'high' ELSE 'low'  pd.cut() or np.where() or .map()
  ROW_NUMBER() OVER (PARTITION BY city)  df.groupby('city').cumcount()
  LAG(col) OVER (ORDER BY date)          df['col'].shift(1)
  SUM() OVER (PARTITION BY user_id)      df.groupby('user_id')['col'].cumsum()
  NTILE(4)                               pd.qcut(df['col'], q=4)
  COALESCE(col, default)                 df['col'].fillna(default)
  COUNT(DISTINCT user_id)                df['user_id'].nunique()
""")

conn.close()
print("All sections complete.")
