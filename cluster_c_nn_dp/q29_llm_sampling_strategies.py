"""
Q29 — LLM Sampling Strategies [ACTUAL INTERVIEW QUESTION — Jun 2026]
Source: Expedia Group | Tech Screen | Senior ML Scientist
Topic:  Temperature scaling, nucleus (top-p) filtering, sample_next_token

WHAT THIS TESTS:
  Do you understand how LLMs actually generate tokens in production?
  Not "what is a transformer" — but: what happens AFTER the model produces logits?

THE PIPELINE (order is critical — explain this out loud first):
  Raw Logits
      ↓  ÷ Temperature      ← apply BEFORE softmax (changes the gaps between scores)
      ↓  Softmax            ← converts to valid probability distribution (sums to 1)
      ↓  Top-p Filter       ← needs valid probs to compute cumulative mass
      ↓  Sample             ← draw one token index

WHY ORDER MATTERS:
  Temperature on logits → scales the GAPS between scores
    e.g. logits [4, 2] → gap is 2. At T=0.5: [8, 4] → gap is 4 → sharper
    If you applied temperature to probabilities it would not have this effect

  Top-p on probabilities → needs a valid distribution (sums to 1) to
    compute "cumulative mass". Can't do cumulative sum on raw logits meaningfully.

VOCAB (40 tokens, travel domain):
  0-3:   <pad> <bos> <eos> <unk>
  4-7:   book flight hotel car
  8-11:  to from in at
  12-19: Paris London Tokyo New York Sydney Dubai Rome
  20-23: a the for on
  24-27: cheap luxury direct roundtrip
  28-31: cancel change upgrade refund
  32-35: seat meal baggage lounge
  36-39: tomorrow weekend January urgent
"""

from __future__ import annotations
import numpy as np

SEED = 42
rng  = np.random.default_rng(SEED)

VOCAB = [
    "<pad>", "<bos>", "<eos>", "<unk>",
    "book", "flight", "hotel", "car",
    "to", "from", "in", "at",
    "Paris", "London", "Tokyo", "New",
    "York", "Sydney", "Dubai", "Rome",
    "a", "the", "for", "on",
    "cheap", "luxury", "direct", "roundtrip",
    "cancel", "change", "upgrade", "refund",
    "seat", "meal", "baggage", "lounge",
    "tomorrow", "weekend", "January", "urgent",
]
VOCAB_SIZE  = len(VOCAB)
TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}

# Synthetic logit arrays (stand-ins for the .npy files)
np.random.seed(0)
# SHARP: model is confident — one or two tokens dominate
LOGITS_SHARP = np.array([5.0 if i in [5, 12] else np.random.uniform(-2, 1)
                         for i in range(VOCAB_SIZE)])
# FLAT: model is uncertain — roughly uniform
LOGITS_FLAT  = np.random.uniform(-0.5, 0.5, VOCAB_SIZE)
# MIXED: realistic mid-confidence
LOGITS_MIXED = np.array([3.0 if i == 5 else 2.0 if i == 12 else np.random.uniform(-1, 1)
                          for i in range(VOCAB_SIZE)])

PROMPT_IDS = [TOKEN_TO_ID["<bos>"], TOKEN_TO_ID["book"],
              TOKEN_TO_ID["a"],      TOKEN_TO_ID["flight"]]


# ══════════════════════════════════════════════════════════════
# SECTION 1: TEMPERATURE SCALING
# ══════════════════════════════════════════════════════════════
#
# CONCEPT:
#   Raw logits: [5.8, 4.2, 3.1, ...]  — model's raw scores before any transformation
#   After ÷T:   [5.8/T, 4.2/T, ...]  — gaps between scores are amplified (T<1) or shrunk (T>1)
#   After softmax: valid probability distribution
#
# INTUITION:
#   T < 1  → divide by small number → scores get BIGGER → gaps get WIDER → sharper dist
#   T = 1  → no change
#   T > 1  → divide by large number → scores get SMALLER → gaps get NARROWER → flatter dist
#   T → 0  → scores → ±∞ → distribution becomes one-hot on argmax (greedy)
#   T → ∞  → scores → 0 → distribution becomes uniform (random)
#
# WHY APPLY TO LOGITS NOT PROBABILITIES?
#   Probabilities are already compressed into [0,1]. Dividing them changes the
#   distribution in a mathematically inconsistent way — it doesn't correspond to
#   any meaningful change in the model's "confidence".
#   Temperature on logits is principled: it scales the unnormalised log-probabilities,
#   which is equivalent to raising each probability to the power 1/T before renormalising.


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.

    FORMULA: softmax(x_i) = exp(x_i) / sum(exp(x_j))

    STABILITY TRICK: subtract max before exp
      Without it: exp(1000) → overflow (inf)
      With it:    exp(1000 - 1000) = exp(0) = 1 → safe
      Mathematical proof it's equivalent:
        exp(x_i - max) / sum(exp(x_j - max))
        = exp(x_i) * exp(-max) / (sum(exp(x_j)) * exp(-max))
        = exp(x_i) / sum(exp(x_j))   ← same result, no overflow

    Args:
        logits: 1-D array of raw scores, shape (vocab_size,)
    Returns:
        Probability distribution that sums to 1.0
    """
    shifted = logits - logits.max()   # subtract max for numerical stability
    exps    = np.exp(shifted)
    return exps / exps.sum()          # normalise to sum = 1


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Scale logits by temperature BEFORE softmax.

    FORMULA: scaled_logits = logits / temperature

    Args:
        logits:      1-D array of raw scores, shape (vocab_size,)
        temperature: Positive float. <1 sharpens; >1 flattens; 0 → greedy
    Returns:
        Scaled logits (same shape). Do NOT apply softmax here.
    Raises:
        ValueError: if temperature <= 0
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / temperature       # one line — that's the whole function


# ── Section 1 checkpoint answers ──────────────────────────────
CHECKPOINT_1 = """
Q1. At temperature=0.3, the top token probability jumped significantly. Why?
A:  Dividing by 0.3 is multiplying by 3.33. If the top logit was 5.0 and the
    second was 3.0, the gap was 2.0. After ÷0.3 the gap becomes 6.67. Softmax
    amplifies gaps exponentially (exp(6.67) >> exp(3.33)), so the top token
    dominates far more. The effect is exponential, not linear.

Q2. At temperature=2.0, the distribution flattened. What is the risk?
A:  The model becomes exploratory to the point of incoherence. A low-probability
    token like "urgent" can get sampled in the middle of "book a flight to Paris".
    In production this manifests as hallucinations, off-topic words, and
    grammatically incoherent sentences.

Q3. Why must temperature be applied to logits, not to softmax output?
A:  Two reasons:
    1. Mathematical: dividing probabilities by T doesn't correspond to any
       principled operation on the model's beliefs. Temperature on logits is
       equivalent to raising each probability to the power 1/T before renormalising
       — a well-defined transformation.
    2. Practical: softmax probabilities are already in [0,1]. Dividing them
       changes their relative ordering inconsistently and you still need to
       renormalise, so you've just added a useless step. Temperature on logits
       changes the gaps in log-probability space, which is where the model
       actually operates.
"""


# ── Section 1 tests ────────────────────────────────────────────
def test_section1():
    probs = softmax(LOGITS_SHARP)
    assert abs(probs.sum() - 1.0) < 1e-5,  "softmax must sum to 1.0"
    assert (probs >= 0).all(),              "softmax outputs must be non-negative"

    scaled = apply_temperature(LOGITS_SHARP, 1.0)
    assert np.allclose(scaled, LOGITS_SHARP), "temperature=1.0 must leave logits unchanged"

    probs_sharp = softmax(apply_temperature(LOGITS_SHARP, 0.5))
    probs_base  = softmax(LOGITS_SHARP)
    assert probs_sharp.max() > probs_base.max(), "lower T must increase top token prob"

    probs_flat = softmax(apply_temperature(LOGITS_SHARP, 2.0))
    assert probs_flat.max() < probs_base.max(),  "higher T must decrease top token prob"

    try:
        apply_temperature(LOGITS_SHARP, 0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass

    print("✅ All Section 1 tests passed.")


# ══════════════════════════════════════════════════════════════
# SECTION 2: NUCLEUS (TOP-P) FILTERING
# ══════════════════════════════════════════════════════════════
#
# PROBLEM: even after temperature scaling, low-probability tokens CAN be sampled.
#   e.g. "urgent" has 0.001 probability — tiny but non-zero. Over thousands of
#   generated tokens, you will eventually sample it at the wrong moment.
#
# SOLUTION: top-p (nucleus) sampling cuts the probability tail.
#
# ALGORITHM:
#   sorted probs (desc): [0.50, 0.25, 0.15, 0.07, 0.03]
#   cumulative sum:      [0.50, 0.75, 0.90, 0.97, 1.00]
#                                       ↑
#   p=0.90 → keep the first 3 tokens (they cover 90% of the mass)
#   zero out tokens 4 and 5
#   renormalise the kept 3 to sum to 1
#
# KEY NUANCE — "include the token that pushes cumsum OVER p":
#   cumsum before token 3 = 0.75 < 0.9 → include token 3
#   After including token 3, cumsum = 0.90 >= 0.9 → stop
#   This ensures you never end up with an empty nucleus.
#
# TOP-P vs TOP-K:
#   top-k: always keep exactly k tokens regardless of their probabilities
#   top-p: keeps a variable number — more tokens when distribution is flat,
#          fewer when model is confident
#   Example where they differ dramatically:
#     Flat distribution (40 equal probs): top-k=5 keeps 5; top-p=0.9 keeps ~36
#     Sharp distribution (one token=0.95): top-k=5 keeps 5; top-p=0.9 keeps 1
#   Top-p is adaptive; top-k is rigid.
#
# WHY RENORMALISE?
#   After zeroing tail tokens, the remaining probs sum to < 1 (we removed mass).
#   numpy's rng.choice needs a valid probability vector (sums to exactly 1).
#   Without renormalisation, sampling would be wrong or crash.


def apply_top_p(probs: np.ndarray, p: float) -> np.ndarray:
    """
    Apply nucleus (top-p) filtering to a probability distribution.

    Args:
        probs: 1-D array of probabilities summing to 1.0, shape (vocab_size,)
        p:     Cumulative probability threshold in (0, 1]. p=1.0 keeps all tokens.
    Returns:
        Re-normalised probability distribution with tail tokens zeroed out.
    """
    # Step 1: sort token indices by probability, highest first
    sorted_indices = np.argsort(probs)[::-1]          # indices of tokens sorted desc by prob
    sorted_probs   = probs[sorted_indices]             # probabilities in that order

    # Step 2: cumulative sum of sorted probabilities
    cumsum = np.cumsum(sorted_probs)

    # Step 3: find the cutoff — keep tokens where cumsum of PREVIOUS tokens < p
    # i.e., include the token that pushes cumsum over p (never empty nucleus)
    # cumsum shifted right by 1: [0, sorted_probs[0], sorted_probs[0]+sorted_probs[1], ...]
    # A token is in the nucleus if its *start* cumsum < p
    cutoff_mask = np.concatenate([[True],              # always keep the top token
                                   (cumsum[:-1] < p)]) # keep while cumsum hasn't reached p yet

    # Step 4: zero out tokens not in the nucleus
    filtered = np.zeros_like(probs)
    filtered[sorted_indices[cutoff_mask]] = sorted_probs[cutoff_mask]

    # Step 5: renormalise so the filtered distribution sums to 1
    total = filtered.sum()
    if total <= 0:
        filtered[sorted_indices[0]] = 1.0             # fallback: keep only argmax
    else:
        filtered /= total                             # divide by remaining mass

    return filtered


# ── Section 2 checkpoint answers ──────────────────────────────
CHECKPOINT_2 = """
Q1. With p=0.5 applied to the mixed distribution, how many tokens remain?
A:  Depends on the distribution. For MIXED_LOGITS (2-3 dominant tokens), roughly
    2-4 tokens. That seems right — we're saying "only sample from the top 50%
    of the probability mass", which for a mid-confidence distribution covers a
    handful of tokens.

Q2. What would happen if you forgot to renormalise after zeroing out tokens?
A:  The distribution would sum to less than 1.0 (e.g., 0.75 if you removed
    25% tail mass). numpy's rng.choice would either crash (probabilities don't
    sum to 1) or silently produce incorrect samples. Renormalising redistributes
    the removed tail mass proportionally among the nucleus tokens.

Q3. Describe a distribution where top-p and top-k give very different results.
A:  Flat distribution: 40 tokens each with prob 0.025.
      top-k=5 keeps exactly 5 tokens
      top-p=0.9 keeps 36 tokens (need 36 × 0.025 = 0.9 to cover 90% mass)

    Sharp distribution: token "flight" has prob 0.95, others share 0.05.
      top-k=5 keeps 5 tokens (including 4 irrelevant ones)
      top-p=0.9 keeps 1 token (just "flight" already covers 95% mass)

    Top-p is adaptive to the model's confidence; top-k is a fixed budget.
"""


# ── Section 2 tests ────────────────────────────────────────────
def test_section2():
    base_probs = softmax(LOGITS_MIXED)

    filtered_all = apply_top_p(base_probs, 1.0)
    assert np.sum(filtered_all > 0) == VOCAB_SIZE, "p=1.0 must keep all tokens"
    assert abs(filtered_all.sum() - 1.0) < 1e-5,  "must sum to 1.0"

    filtered_tight = apply_top_p(base_probs, 0.5)
    filtered_loose  = apply_top_p(base_probs, 0.9)
    assert np.sum(filtered_tight > 0) < np.sum(filtered_loose > 0), \
        "smaller p must keep fewer tokens"

    for p_val in [0.3, 0.5, 0.9, 1.0]:
        out = apply_top_p(base_probs, p_val)
        assert abs(out.sum() - 1.0) < 1e-5, f"must sum to 1.0 at p={p_val}"

    filtered_tiny = apply_top_p(base_probs, 0.01)
    assert np.sum(filtered_tiny > 0) >= 1, "at least one token must always survive"

    print("✅ All Section 2 tests passed.")


# ══════════════════════════════════════════════════════════════
# SECTION 3: sample_next_token
# ══════════════════════════════════════════════════════════════
#
# WIRING THE PIPELINE TOGETHER
#
# EXACT ORDER (say this out loud in the interview):
#   1. temperature=0?  → return argmax immediately (greedy, no sampling)
#   2. apply_temperature(logits, T)   → scale logits
#   3. softmax(scaled_logits)         → convert to probabilities
#   4. apply_top_p(probs, top_p)      → filter tail
#   5. rng.choice(vocab_size, p=filtered_probs)  → sample one token
#
# WHY TEMPERATURE BEFORE SOFTMAX:
#   Temperature scales the LOG-probability space. After softmax you're in
#   probability space — dividing probabilities by T has no principled meaning
#   and would change relative rankings inconsistently.
#
# WHY TOP-P AFTER SOFTMAX:
#   Top-p computes a CUMULATIVE SUM of probabilities. You need the values to
#   sum to 1 first. Computing cumsum of raw logits is meaningless (logits can
#   be negative, don't sum to 1, can't represent a nucleus).
#
# GREEDY (temperature=0):
#   As T→0, scaled_logits → ±∞, softmax → one-hot on argmax.
#   We handle T=0 as a special case to avoid division by zero.
#   Return argmax(logits) directly.


def sample_next_token(
    logits:      np.ndarray,
    temperature: float = 1.0,
    top_p:       float = 1.0,
    rng:         np.random.Generator = np.random.default_rng(0),
) -> int:
    """
    Sample one token index from logits using temperature + top-p.

    Pipeline: logits → ÷T → softmax → top-p filter → sample

    Args:
        logits:      1-D array of raw model scores, shape (vocab_size,)
        temperature: Sampling temperature. 0 = greedy (argmax).
        top_p:       Nucleus probability threshold. 1.0 = keep all tokens.
        rng:         NumPy random generator (pass explicitly for reproducibility).
    Returns:
        Single integer token index sampled from the filtered distribution.
    """
    # Special case: temperature=0 → greedy decoding (deterministic argmax)
    # Handles the mathematical limit T→0 without division by zero
    if temperature == 0.0:
        return int(np.argmax(logits))

    # Step 1: temperature scaling (on logits, BEFORE softmax)
    scaled = apply_temperature(logits, temperature)

    # Step 2: softmax → valid probability distribution
    probs = softmax(scaled)

    # Step 3: top-p filtering (on probabilities, AFTER softmax)
    filtered = apply_top_p(probs, top_p)

    # Step 4: sample one token index from the filtered distribution
    return int(rng.choice(len(filtered), p=filtered))


# ── Section 3 checkpoint answers ──────────────────────────────
CHECKPOINT_3 = """
Q1. What breaks if you apply softmax BEFORE temperature?
A:  You're now dividing probabilities by T, not logits. The mathematics no longer
    correspond to scaling log-probabilities. Concretely:
    - softmax([5, 3]) = [0.88, 0.12]. Dividing by T=0.5 gives [1.76, 0.24].
      These don't sum to 1 — you'd need to renormalise again, and the result
      is NOT the same as softmax([10, 6]) (what T=0.5 on logits gives).
    - The relative ordering of probabilities is preserved but the shape is wrong.
      You lose the principled connection between temperature and model confidence.

Q2. What breaks if you apply top-p to raw logits instead of probabilities?
A:  Top-p is defined in terms of cumulative probability mass. Raw logits:
    - Can be negative (cumulative sum of negatives is meaningless as "mass")
    - Don't sum to 1 — you can't define "90% of the mass" on them
    - The cumulative sum would grow without bound for large vocabs
    The algorithm would produce nonsensical cutoffs. At minimum it would crash;
    at worst it would silently keep the wrong set of tokens.

Q3. When would you set temperature=0 in production? Gain and lose what?
A:  USE temperature=0 for:
    - Deterministic tasks: intent classification, SQL generation, structured JSON
      extraction — where the correct answer is singular and reproducibility matters
    - Caching: same input always produces same output → safe to cache
    - Evaluation: comparing model outputs fairly requires reproducibility

    GAIN: determinism, reproducibility, highest-probability token every time
    LOSE: diversity, creativity, ability to avoid bad outputs (if argmax is wrong,
          you're stuck — there's no randomness to recover from a local maximum)

    Failure example: a customer apology email with temperature=0 will produce
    the same boilerplate every time. If that boilerplate is slightly wrong
    (e.g., wrong refund amount), every single customer gets the wrong email.
    Diversity would have caught the edge cases.
"""


# ── Section 3 tests ────────────────────────────────────────────
def test_section3():
    # Test 1: greedy always returns the same token (argmax)
    greedy_tokens = {
        sample_next_token(LOGITS_MIXED, temperature=0.0, rng=np.random.default_rng(i))
        for i in range(10)
    }
    assert len(greedy_tokens) == 1,       "temperature=0 must always return same token"
    assert list(greedy_tokens)[0] == int(np.argmax(LOGITS_MIXED)), "must return argmax"

    # Test 2: sampled token is a valid vocab index
    token = sample_next_token(LOGITS_MIXED, temperature=1.0, top_p=0.9,
                              rng=np.random.default_rng(0))
    assert 0 <= token < VOCAB_SIZE, "sampled token must be a valid vocab index"

    # Test 3: higher temperature → more variety in sampled tokens
    low_temp_tokens  = {sample_next_token(LOGITS_MIXED, temperature=0.3, top_p=1.0,
                                          rng=np.random.default_rng(i)) for i in range(20)}
    high_temp_tokens = {sample_next_token(LOGITS_MIXED, temperature=2.0, top_p=1.0,
                                          rng=np.random.default_rng(i)) for i in range(20)}
    assert len(high_temp_tokens) >= len(low_temp_tokens), \
        "higher temperature must produce more token variety"

    print("✅ All Section 3 tests passed.")


# ══════════════════════════════════════════════════════════════
# SECTION 4: JUDGMENT — SAMPLING IN PRODUCTION
# ══════════════════════════════════════════════════════════════
#
# Three tasks, three very different sampling configurations.
# The key question: how much diversity do you want vs how predictable must the output be?

SECTION_4_ANSWERS = """
╔══════════════════════════════════════════════════════════════════════╗
║  Task A — Intent Labeling (500 queries → 1 of 8 labels)             ║
╚══════════════════════════════════════════════════════════════════════╝

  temperature = 0  (or 0.1 at most)
  top_p       = 1.0 (irrelevant with T=0, but explicitly set for clarity)

  WHY:
    The correct answer is singular — "cancel_flight" or "book_hotel", not
    a creative interpretation. You want the highest-probability label every time.
    These labels feed model training: variability introduces noise into your
    ground truth, which corrupts the downstream model.
    Determinism also means the pipeline is auditable — same input = same label.

  WHAT BREAKS WITH WRONG SETTINGS:
    temperature=1.5 → "book hotel in Tokyo" might get labelled "refund_request"
    3% of the time when the model is uncertain. Over 500 queries that's 15 wrong
    labels injected into your training data. Your downstream model learns garbage.

╔══════════════════════════════════════════════════════════════════════╗
║  Task B — Customer Apology Email (flight cancellation refund)        ║
╚══════════════════════════════════════════════════════════════════════╝

  temperature = 0.7
  top_p       = 0.9

  WHY:
    The email must be empathetic and professional (needs some natural language
    variation — same exact phrasing every time sounds robotic) BUT must also
    be factually accurate (refund amount, policy details).
    T=0.7 produces coherent text with slight variation in tone and phrasing.
    top_p=0.9 prevents absurd tail tokens ("urgent" mid-apology) while
    allowing multiple natural phrasings of the same message.

  WHAT BREAKS WITH WRONG SETTINGS:
    temperature=0 → every customer gets identical boilerplate. If the
    template says "3-5 business days" but actual SLA is 7 days, every
    single customer gets wrong information.

    temperature=2.0 → "Dear valued customer, your refund of $undefined has been
    processed with urgent baggage lounge at the Paris seat tomorrow." Incoherent.
    Customer escalates. Brand damage.

╔══════════════════════════════════════════════════════════════════════╗
║  Task C — Itinerary Suggestions ("surprise me" — 4 trip ideas)       ║
╚══════════════════════════════════════════════════════════════════════╝

  temperature = 1.2 – 1.5
  top_p       = 0.95

  WHY:
    The user explicitly wants variety and surprise. Deterministic generation
    would produce the same 4 cities every time for every user — useless.
    Higher temperature forces the model to explore lower-probability options
    (Tokyo instead of always Paris, weekend hiking instead of always beach).
    top_p=0.95 still prevents truly off-the-wall tokens while allowing
    creative combinations.
    If generating 4 separate ideas, you might also use different random seeds
    per call to guarantee diversity across the 4 suggestions.

  WHAT BREAKS WITH WRONG SETTINGS:
    temperature=0 → every user who says "surprise me" gets the same 4 suggestions:
    Paris, London, Tokyo, New York. No surprise. Users churn.

    temperature=3.0 → "Weekend trip idea 1: urgent lounge roundtrip the."
    Completely incoherent. top_p=0.95 helps but very high T overrides it.

SUMMARY TABLE:
  Task           T        top_p   Why
  ─────────────────────────────────────────────────────────────────
  Intent label   0.0      1.0     Single correct answer, feeds training
  Apology email  0.7      0.9     Coherent but natural, filter tail tokens
  Surprise me    1.2-1.5  0.95    Diversity required, slight tail filter

GENERAL RULE:
  Higher stakes + singular correct answer → lower temperature
  Creative output + diversity required   → higher temperature
  Always use top_p < 1.0 in production  → prevents degenerate tail tokens
"""


# ══════════════════════════════════════════════════════════════
# RUN EVERYTHING
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("SECTION 1: Temperature Scaling")
    print("=" * 60)

    temps = [0.3, 1.0, 2.0]
    for t in temps:
        scaled = apply_temperature(LOGITS_SHARP, t)
        probs  = softmax(scaled)
        top5   = [(ID_TO_TOKEN[i], round(float(probs[i]), 4))
                  for i in np.argsort(probs)[::-1][:5]]
        print(f"  T={t:.1f}  top-5: {top5}")

    test_section1()

    print("\n" + "=" * 60)
    print("SECTION 2: Nucleus (Top-p) Filtering")
    print("=" * 60)

    base_probs = softmax(LOGITS_MIXED)
    for p_val in [0.5, 0.9, 1.0]:
        filtered   = apply_top_p(base_probs, p_val)
        n_nonzero  = int(np.sum(filtered > 0))
        top3       = [(ID_TO_TOKEN[i], round(float(filtered[i]), 4))
                      for i in np.argsort(filtered)[::-1][:3]]
        print(f"  p={p_val:.1f}  tokens kept: {n_nonzero:2d}  top-3: {top3}")

    test_section2()

    print("\n" + "=" * 60)
    print("SECTION 3: sample_next_token")
    print("=" * 60)

    print("  Greedy (T=0) — must always be the same token:")
    greedy = [sample_next_token(LOGITS_MIXED, temperature=0.0,
                                rng=np.random.default_rng(i)) for i in range(5)]
    print(f"  {[ID_TO_TOKEN[t] for t in greedy]}")

    print("\n  Sampling (T=1.0, p=0.9) — should vary:")
    samples = [sample_next_token(LOGITS_MIXED, temperature=1.0, top_p=0.9,
                                  rng=np.random.default_rng(i)) for i in range(10)]
    print(f"  {[ID_TO_TOKEN[t] for t in samples]}")

    print("\n  High temperature (T=2.0, p=0.95) — more variety:")
    hot = [sample_next_token(LOGITS_MIXED, temperature=2.0, top_p=0.95,
                              rng=np.random.default_rng(i)) for i in range(10)]
    print(f"  {[ID_TO_TOKEN[t] for t in hot]}")

    test_section3()

    print("\n" + "=" * 60)
    print("SECTION 4: Sampling in Production")
    print("=" * 60)
    print(SECTION_4_ANSWERS)

    print("\n" + "=" * 60)
    print("SECTION 1 CHECKPOINT ANSWERS")
    print("=" * 60)
    print(CHECKPOINT_1)

    print("=" * 60)
    print("SECTION 2 CHECKPOINT ANSWERS")
    print("=" * 60)
    print(CHECKPOINT_2)

    print("=" * 60)
    print("SECTION 3 CHECKPOINT ANSWERS")
    print("=" * 60)
    print(CHECKPOINT_3)

    print("\n✅ All sections complete.")
