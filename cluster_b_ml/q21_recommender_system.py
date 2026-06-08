"""
Q21 — Recommender System [HIGH PROBABILITY]
Target time: 25 min | Requires: numpy, pandas, scikit-learn

APPROACH (say this in the first 60 seconds):
"I'll build a two-stage recommender: Stage 1 is candidate generation using
collaborative filtering (matrix factorization) plus content-based fallback
for cold-start. Stage 2 is ranking with a simple scoring model. I'll evaluate
with NDCG@k. In production this becomes a two-tower retrieval + LambdaMART
ranking pipeline."

CORE CONCEPTS:
  Collaborative Filtering: "users who liked X also liked Y"
    → Matrix Factorization: decompose R (user x item) into U * V^T
    → SVD or ALS (Alternating Least Squares)

  Content-Based: match item features to user preference profile
    → Good for cold-start (new items, new users)

  Hybrid: combine both signals

RANKING METRICS:
  NDCG@k = DCG@k / IDCG@k
    DCG@k  = sum_{i=1}^{k} rel_i / log2(i+1)   # reward relevance near top
    IDCG@k = DCG of ideal (perfectly sorted) ranking

  MAP@k  = mean average precision across users (binary relevance)
  MRR    = mean reciprocal rank of first relevant item
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────────────────────
# SECTION 1: Synthetic data — user-item interaction matrix
# ─────────────────────────────────────────────────────────────

np.random.seed(42)
N_USERS, N_ITEMS = 50, 20

# Ratings matrix: 0 = not interacted, 1-5 = rating
ratings = np.random.choice(
    [0, 0, 0, 1, 2, 3, 4, 5],       # sparse: ~37% fill
    size=(N_USERS, N_ITEMS)
).astype(float)

# Item features (price tier, avg review, category one-hot)
item_features = pd.DataFrame({
    "price_tier":   np.random.randint(1, 4, N_ITEMS),    # 1=budget, 3=luxury
    "avg_review":   np.round(np.random.uniform(3.0, 5.0, N_ITEMS), 1),
    "is_hotel":     np.random.randint(0, 2, N_ITEMS),
    "is_apartment": np.random.randint(0, 2, N_ITEMS),
    "central_loc":  np.random.randint(0, 2, N_ITEMS),
})

user_ids = [f"user_{i}" for i in range(N_USERS)]
item_ids = [f"item_{i}" for i in range(N_ITEMS)]


# ─────────────────────────────────────────────────────────────
# SECTION 2: Collaborative Filtering (User-Based)
# ─────────────────────────────────────────────────────────────

def user_based_cf(ratings_matrix: np.ndarray, target_user: int, top_k: int = 5) -> list:
    """
    User-based collaborative filtering.
    1. Compute cosine similarity between target user and all others.
    2. Weight other users' ratings by similarity.
    3. Return top-k unrated items.
    """
    # Mean-centre ratings (only non-zero entries)
    centred = ratings_matrix.copy()
    for u in range(ratings_matrix.shape[0]):
        rated = centred[u] > 0
        if rated.any():
            centred[u, rated] -= centred[u, rated].mean()

    # Similarity between all user pairs
    user_sim = cosine_similarity(centred)  # (N_USERS, N_USERS)
    sims = user_sim[target_user]           # similarity to target

    # Predict scores for unrated items
    already_rated = ratings_matrix[target_user] > 0
    scores = {}
    for item in range(ratings_matrix.shape[1]):
        if already_rated[item]:
            continue
        # weighted average over users who rated this item
        rated_by = ratings_matrix[:, item] > 0
        if not rated_by.any():
            continue
        numerator   = np.dot(sims[rated_by], ratings_matrix[rated_by, item])
        denominator = np.sum(np.abs(sims[rated_by])) + 1e-9
        scores[item] = numerator / denominator

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:top_k]


# ─────────────────────────────────────────────────────────────
# SECTION 3: Content-Based Filtering (cold-start fallback)
# ─────────────────────────────────────────────────────────────

def content_based(
    ratings_matrix: np.ndarray,
    item_feat: pd.DataFrame,
    target_user: int,
    top_k: int = 5
) -> list:
    """
    Build a user preference profile from rated items' features.
    Score unrated items by cosine similarity to that profile.
    """
    scaler = MinMaxScaler()
    feat_scaled = scaler.fit_transform(item_feat.values)  # (N_ITEMS, n_features)

    rated_mask  = ratings_matrix[target_user] > 0
    rated_idx   = np.where(rated_mask)[0]

    if len(rated_idx) == 0:
        # True cold-start: return most popular items
        popularity = (ratings_matrix > 0).sum(axis=0)
        return sorted(enumerate(popularity), key=lambda x: -x[1])[:top_k]

    # User profile = weighted average of rated item features
    weights    = ratings_matrix[target_user, rated_idx]
    user_profile = np.average(feat_scaled[rated_idx], axis=0, weights=weights)

    scores = {}
    for item in range(ratings_matrix.shape[1]):
        if rated_mask[item]:
            continue
        sim = cosine_similarity([user_profile], [feat_scaled[item]])[0, 0]
        scores[item] = float(sim)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:top_k]


# ─────────────────────────────────────────────────────────────
# SECTION 4: Hybrid — combine CF + content scores
# ─────────────────────────────────────────────────────────────

def hybrid_recommend(
    ratings_matrix: np.ndarray,
    item_feat: pd.DataFrame,
    target_user: int,
    alpha: float = 0.7,
    top_k: int = 5
) -> list:
    """
    alpha * CF_score + (1 - alpha) * content_score.
    alpha=0.7 weights CF higher (good when enough interactions exist).
    Fall back to content-only if user has fewer than 3 ratings.
    """
    n_rated = (ratings_matrix[target_user] > 0).sum()
    if n_rated < 3:
        alpha = 0.0  # pure content for cold-start users

    cf_raw      = dict(user_based_cf(ratings_matrix, target_user, top_k=N_ITEMS))
    content_raw = dict(content_based(ratings_matrix, item_feat, target_user, top_k=N_ITEMS))

    # Normalise each to [0, 1]
    def normalise(d: dict) -> dict:
        if not d:
            return d
        mn, mx = min(d.values()), max(d.values())
        if mx == mn:
            return {k: 1.0 for k in d}
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}

    cf_norm      = normalise(cf_raw)
    content_norm = normalise(content_raw)

    all_items = set(cf_norm) | set(content_norm)
    hybrid_scores = {
        item: alpha * cf_norm.get(item, 0) + (1 - alpha) * content_norm.get(item, 0)
        for item in all_items
    }
    ranked = sorted(hybrid_scores.items(), key=lambda x: -x[1])
    return ranked[:top_k]


# ─────────────────────────────────────────────────────────────
# SECTION 5: Evaluation — NDCG@k
# ─────────────────────────────────────────────────────────────

def dcg_at_k(relevances: list, k: int) -> float:
    """DCG@k — higher weight for relevant items near the top."""
    relevances = relevances[:k]
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(recommended: list, relevant_items: set, k: int) -> float:
    """
    NDCG@k.
    recommended : ordered list of item ids
    relevant_items: set of truly relevant item ids (e.g., held-out positives)
    """
    rel = [1 if item in relevant_items else 0 for item in recommended[:k]]
    ideal = sorted(rel, reverse=True)
    dcg   = dcg_at_k(rel, k)
    idcg  = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommender(ratings_matrix: np.ndarray, item_feat: pd.DataFrame, k: int = 5) -> float:
    """
    Leave-one-out evaluation:
    For each user, hide their highest-rated item, run recommender,
    check if the hidden item appears in top-k.
    """
    ndcg_scores = []
    for user in range(ratings_matrix.shape[0]):
        rated = np.where(ratings_matrix[user] > 0)[0]
        if len(rated) < 2:
            continue
        # Hold out the item with the highest rating
        held_out = rated[np.argmax(ratings_matrix[user, rated])]
        train = ratings_matrix.copy()
        train[user, held_out] = 0

        recs = hybrid_recommend(train, item_feat, user, top_k=k)
        rec_items = [r[0] for r in recs]
        ndcg = ndcg_at_k(rec_items, {held_out}, k)
        ndcg_scores.append(ndcg)

    return float(np.mean(ndcg_scores))


# ─────────────────────────────────────────────────────────────
# SECTION 6: Run everything
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    target = 0  # evaluate for user_0

    print("=" * 60)
    print("SECTION 2: User-Based Collaborative Filtering")
    print("=" * 60)
    cf_recs = user_based_cf(ratings, target, top_k=5)
    print(f"\nTop-5 CF recommendations for {user_ids[target]}:")
    for item, score in cf_recs:
        print(f"  {item_ids[item]:10s}  score={score:.4f}")

    print("\n" + "=" * 60)
    print("SECTION 3: Content-Based Filtering")
    print("=" * 60)
    cb_recs = content_based(ratings, item_features, target, top_k=5)
    print(f"\nTop-5 content-based recommendations for {user_ids[target]}:")
    for item, score in cb_recs:
        print(f"  {item_ids[item]:10s}  score={score:.4f}")

    print("\n" + "=" * 60)
    print("SECTION 4: Hybrid Recommendations")
    print("=" * 60)
    hybrid_recs = hybrid_recommend(ratings, item_features, target, top_k=5)
    print(f"\nTop-5 hybrid recommendations for {user_ids[target]}:")
    for item, score in hybrid_recs:
        print(f"  {item_ids[item]:10s}  score={score:.4f}")

    print("\n" + "=" * 60)
    print("SECTION 5: NDCG@5 Evaluation (leave-one-out)")
    print("=" * 60)
    ndcg = evaluate_recommender(ratings, item_features, k=5)
    print(f"\nMean NDCG@5 across all users: {ndcg:.4f}")
    print("(Random baseline ≈ 0.20 for k=5; good models reach 0.50+)")

    # ── Quick sanity checks ──
    # item 0 at position 0 → perfect NDCG
    assert abs(ndcg_at_k([0, 1, 2], {0}, k=3) - 1.0) < 1e-9
    # item 0 at position 2, item 4 at position 5 → partial credit
    assert ndcg_at_k([0, 1, 0, 0, 1], {1, 4}, k=5) > 0
    # no relevant item → 0
    assert ndcg_at_k([5, 6, 7], {99}, k=3) == 0.0
    print("\nAll sanity checks passed.")

    print("\n" + "=" * 60)
    print("KEY TALKING POINTS")
    print("=" * 60)
    print("""
Two-stage architecture:
  Stage 1 — Candidate generation (~1k candidates, cheap)
    • CF (matrix factorization / two-tower NN)
    • Content-based (location, price, amenities)
    • Popularity fallback for cold-start

  Stage 2 — Ranking (1k → 20, expensive features OK)
    • LambdaMART / LightGBM optimising NDCG@10
    • User × item × context features
    • Re-rank for diversity + business rules

Cold-start:
  New user  → content-based + popularity (no interaction history)
  New item  → feature-based scoring + impression floor budget

Evaluation:
  Offline: NDCG@k, MAP, Recall@k  (logged data, A/B holdout)
  Online:  CTR, booking conversion, revenue per search

Feedback loop:
  Clicks / bookings / dwell time → implicit labels → retrain ranker
""")
