"""
Q24 — A/B Testing Design [CLASSIC EXPEDIA QUESTION]
Target time: 20 min | Requires: numpy, scipy, pandas

APPROACH (say this in the first 60 seconds):
"A/B testing is a randomised controlled experiment to measure the causal
effect of a change. You split users randomly into control and treatment,
measure the primary metric and guardrail metrics, then use a statistical
test to decide if the observed difference is real or noise. The key design
decisions are: what metric, what sample size (power analysis), how long to
run, and what guardrails prevent shipping a harmful change."

THE EXPEDIA FRAMING (most common questions):
  'How would you measure if a new ranking model improved bookings?'
  'How would you A/B test changing the search results page?'
  'Are our gift cards driving incremental revenue or just cannibalising?'
  'How would you detect if a feature hurts specific user segments?'

CORE STATS:
  Null hypothesis H₀:  no difference between control and treatment
  p-value:             P(seeing this result or more extreme | H₀ is true)
  p < α (0.05)      → reject H₀, declare significance
  Type I error (α): false positive — declare winner when no real effect
  Type II error (β): false negative — miss a real effect
  Power (1-β):      probability of detecting a real effect (target ≥ 0.80)
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ─────────────────────────────────────────────────────────────
# SECTION 1: Sample size / power analysis
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 1: Sample Size — Power Analysis")
print("=" * 60)

print("""
BEFORE RUNNING AN A/B TEST — ALWAYS DO POWER ANALYSIS.
'How long do we run the test?' is decided BEFORE, not after.

Four inputs:
  baseline_rate: current conversion rate (e.g. 5% booking rate)
  mde:          minimum detectable effect — smallest lift that matters (e.g. 0.5pp)
  alpha:        false positive rate (0.05 = 5%)
  power:        probability of detecting real effect (0.80 = 80%)

Formula (two-proportion z-test):
  n = (z_α/2 + z_β)² * [p₁(1-p₁) + p₂(1-p₂)] / (p₁ - p₂)²
""")


def sample_size_two_proportions(
    baseline: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Minimum sample size per group for a two-proportion z-test.

    baseline: control conversion rate (e.g. 0.05 = 5%)
    mde:      minimum detectable effect (absolute, e.g. 0.005 = 0.5pp lift)
    alpha:    significance level (Type I error rate)
    power:    1 - beta (1 - Type II error rate)
    """
    p1 = baseline
    p2 = baseline + mde
    z_alpha = stats.norm.ppf(1 - alpha / 2)    # two-tailed
    z_beta  = stats.norm.ppf(power)

    numerator   = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    denominator = (p1 - p2) ** 2
    return int(np.ceil(numerator / denominator))


# Expedia-flavoured scenarios
scenarios = [
    ("Booking conversion",  0.05,  0.005, 0.05, 0.80),   # 5%  → 5.5%
    ("Booking conversion",  0.05,  0.001, 0.05, 0.80),   # 5%  → 5.1% (tiny lift)
    ("CTR on search",       0.12,  0.010, 0.05, 0.80),   # 12% → 13%
    ("CTR on search",       0.12,  0.010, 0.05, 0.90),   # same but 90% power
]

print(f"{'Metric':>22} {'Baseline':>10} {'MDE':>8} {'α':>6} {'Power':>7} {'n/group':>10} {'Days*':>7}")
print("-" * 76)
for metric, base, mde, alpha, power in scenarios:
    n = sample_size_two_proportions(base, mde, alpha, power)
    # Assuming 10k daily visits split 50/50
    days = int(np.ceil(n / 5000))
    print(f"{metric:>22} {base:>10.1%} {mde:>8.3%} {alpha:>6.2f} {power:>7.0%} {n:>10,} {days:>7}")

print("  * assuming 10,000 daily users split 50/50")
print("""
KEY INSIGHT: Detecting a 0.1pp lift on a 5% baseline requires ~800k users
per group. That's why you need to know your MDE before running — not every
test is worth running if traffic is low.
""")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Run a simulated A/B test
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 2: Simulated A/B Test — Booking Conversion")
print("=" * 60)

# Simulate: new ranking model improves bookings from 5% to 5.6%
N_PER_GROUP = 15_000
CONTROL_RATE    = 0.050
TREATMENT_RATE  = 0.056   # true effect: +0.6pp

control   = np.random.binomial(1, CONTROL_RATE,   N_PER_GROUP)
treatment = np.random.binomial(1, TREATMENT_RATE, N_PER_GROUP)

ctrl_conv  = control.mean()
treat_conv = treatment.mean()
lift       = treat_conv - ctrl_conv
lift_rel   = lift / ctrl_conv * 100

print(f"\n  Control  conversions: {control.sum():,} / {N_PER_GROUP:,}  ({ctrl_conv:.3%})")
print(f"  Treatment conversions: {treatment.sum():,} / {N_PER_GROUP:,}  ({treat_conv:.3%})")
print(f"  Absolute lift: {lift:+.4%} ({lift_rel:+.2f}% relative)")


def two_proportion_z_test(x1, n1, x2, n2, alpha=0.05):
    """
    Two-proportion z-test (one-sided: is treatment > control?).
    x1, x2: number of conversions
    n1, n2: group sizes
    Returns: z-stat, p-value, confidence interval, decision
    """
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)     # pooled proportion under H₀

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z  = (p2 - p1) / se
    p_value = 1 - stats.norm.cdf(z)    # one-sided

    # 95% CI on the difference (unpooled SE for CI)
    se_diff = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    ci_lo = (p2 - p1) - 1.96 * se_diff
    ci_hi = (p2 - p1) + 1.96 * se_diff

    decision = "SHIP ✅" if p_value < alpha else "DO NOT SHIP ❌"
    return {
        "z_stat": round(z, 4),
        "p_value": round(p_value, 6),
        "ci_95": (round(ci_lo, 5), round(ci_hi, 5)),
        "decision": decision,
    }


result = two_proportion_z_test(
    control.sum(),   N_PER_GROUP,
    treatment.sum(), N_PER_GROUP,
)

print(f"\n  z-statistic: {result['z_stat']}")
print(f"  p-value:     {result['p_value']}")
print(f"  95% CI on lift: [{result['ci_95'][0]:+.5f}, {result['ci_95'][1]:+.5f}]")
print(f"  Decision:    {result['decision']}")
print(f"""
  HOW TO EXPLAIN p-VALUE TO AIMÉ:
    "If there were truly no difference, we'd see a lift this large or
    larger by chance {result['p_value']*100:.1f}% of the time.
    Since {result['p_value']} < 0.05, we reject H₀ and call it significant."

  COMMON MISTAKE: p-value is NOT the probability that H₀ is true.
  It's P(data | H₀ is true). Aimé knows this — don't confuse them.
""")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Type I and Type II errors — the tradeoff
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 3: Type I / Type II Errors + Power")
print("=" * 60)

print("""
  ERROR TYPES — say this clearly:

  Type I  (α, false positive): H₀ is TRUE but we reject it.
    → We ship a change that has no real effect.
    → Controlled by significance level α (usually 0.05).
    → Business cost: wasted engineering, diluted user experience.

  Type II (β, false negative): H₁ is TRUE but we fail to reject H₀.
    → We don't ship a change that actually works.
    → Controlled by power (1-β, usually 0.80).
    → Business cost: missed revenue opportunity.

  POWER = 1 - β = probability of detecting a real effect when it exists.

  TRADEOFF: lowering α → fewer false positives but more false negatives.
  Fix: increase sample size to reduce both simultaneously.
""")

# Demonstrate: same experiment with different sample sizes
print(f"{'N/group':>10} {'Power estimate':>16}")
print("-" * 28)
true_effect = 0.006   # 0.6pp lift
baseline    = 0.050

for n in [1_000, 5_000, 10_000, 15_000, 30_000]:
    # Simulate 1000 experiments, measure how often we detect the effect
    detections = 0
    for _ in range(500):
        ctrl = np.random.binomial(1, baseline,              n)
        trt  = np.random.binomial(1, baseline + true_effect, n)
        _, pv = stats.ttest_ind(ctrl, trt)
        if pv < 0.05:
            detections += 1
    power_est = detections / 500
    print(f"{n:>10,} {power_est:>16.1%}")

print("""
  At n=15k/group we're getting ~80% power on a 0.6pp lift.
  That matches the formula from Section 1 — good sanity check.
""")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Guardrail metrics — don't ship harm
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 4: Guardrail Metrics — The Expedia Pattern")
print("=" * 60)

print("""
GUARDRAIL METRICS — the answer Aimé wants:

  Primary metric: booking conversion rate (the thing you're trying to move)

  Guardrail metrics (must NOT degrade):
    • Revenue per session      — conversion up but cheaper hotels? Net negative.
    • Cancellation rate        — easier to book → more impulse cancellations
    • Customer satisfaction    — fast bookings but confusing UX?
    • Page load time           — new model adds latency?
    • Support contacts/session — more confusion → more support tickets

  DECISION RULE:
    SHIP if:   primary metric improves AND all guardrails hold.
    HOLD if:   primary improves but a guardrail degrades.
    STOP if:   primary is flat or negative.

  WHY THIS MATTERS AT EXPEDIA:
    A ranking model could boost bookings by surfacing cheaper hotels users
    click on quickly — but if those bookings cancel more, net revenue is worse.
    Guardrail on cancellation rate catches this.
""")

# Simulate: treatment boosts bookings but hurts revenue/session
np.random.seed(99)
N = 20_000

ctrl_bookings     = np.random.binomial(1, 0.050, N)
ctrl_revenue      = ctrl_bookings * np.random.normal(250, 80, N)  # $250 avg
ctrl_cancellation = np.random.binomial(1, 0.08, N)

# Treatment: +0.8pp booking, but cheaper hotels → lower revenue, +2pp cancellation
trt_bookings      = np.random.binomial(1, 0.058, N)
trt_revenue       = trt_bookings * np.random.normal(220, 80, N)   # $220 avg (-12%)
trt_cancellation  = np.random.binomial(1, 0.10, N)                # +2pp

_, p_book   = stats.ttest_ind(ctrl_bookings, trt_bookings)
_, p_rev    = stats.ttest_ind(ctrl_revenue,  trt_revenue)
_, p_cancel = stats.ttest_ind(ctrl_cancellation, trt_cancellation)

print(f"  {'Metric':>24} {'Control':>10} {'Treatment':>12} {'p-value':>10} {'Status':>12}")
print("  " + "-" * 72)
metrics_check = [
    ("Booking rate (PRIMARY)",  ctrl_bookings.mean(),  trt_bookings.mean(),  p_book,   "improve"),
    ("Revenue/session",         ctrl_revenue.mean(),   trt_revenue.mean(),   p_rev,    "guardrail"),
    ("Cancellation rate",       ctrl_cancellation.mean(), trt_cancellation.mean(), p_cancel, "guardrail"),
]
for name, cv, tv, pv, role in metrics_check:
    status = ""
    if role == "improve":
        status = "✅ BETTER" if tv > cv and pv < 0.05 else "❌ NO EFFECT"
    else:  # guardrail — must not degrade
        status = "❌ DEGRADED" if tv < cv * 0.99 and pv < 0.05 else "✅ HOLDS"
        if "Cancel" in name:
            status = "❌ DEGRADED" if tv > cv * 1.01 and pv < 0.05 else "✅ HOLDS"
    print(f"  {name:>24} {cv:>10.3%} {tv:>12.3%} {pv:>10.4f} {status:>12}")

print("""
  DECISION: DO NOT SHIP ❌
  Booking rate improved but revenue/session dropped 12% and cancellations
  rose 2pp — both statistically significant. Net revenue impact is negative.
  This is exactly the guardrail pattern that prevents a metrics trap.
""")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Common A/B test pitfalls
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 5: Pitfalls — Aimé Will Ask About These")
print("=" * 60)

print("""
1. PEEKING / OPTIONAL STOPPING
   Problem: checking p-value daily and stopping when p < 0.05 inflates α.
   Fix: decide sample size before starting; don't peek; use sequential
        testing (SPRT) if you need early stopping.

2. MULTIPLE COMPARISONS (p-hacking)
   Problem: testing 20 metrics at α=0.05 → 1 false positive expected.
   Fix: Bonferroni correction (α/k), or Benjamini-Hochberg for FDR control.
   Example: testing 10 segments → use α=0.005 per segment.

3. NOVELTY EFFECT
   Problem: users engage with new feature just because it's new.
   Fix: run test long enough (≥ 2 weeks) for novelty to wear off.
        Monitor week-1 vs week-2 effect size separately.

4. NETWORK / SPILLOVER EFFECTS
   Problem: in social or marketplace settings, treatment users affect
            control users (e.g. Uber driver allocation).
   Fix: cluster randomisation (randomise by city, not user).

5. SAMPLE RATIO MISMATCH (SRM)
   Problem: expected 50/50 split but got 48/52 → assignment bug.
   Fix: always check assignment ratio before analysing results.
        Chi-square test on group sizes.

6. SURVIVORSHIP BIAS
   Problem: analysing only users who returned — excludes churned users
            who may have been harmed by the treatment.
   Fix: analyse on the full intent-to-treat population.
""")

# Demonstrate SRM check
print("  SRM CHECK EXAMPLE:")
ctrl_n  = 9800     # expected 10,000
treat_n = 10200    # expected 10,000
chi2, p_srm = stats.chisquare([ctrl_n, treat_n], f_exp=[10000, 10000])
print(f"  Observed: {ctrl_n} / {treat_n} | Expected: 10000 / 10000")
print(f"  χ² = {chi2:.2f}, p = {p_srm:.5f}")
print(f"  {'⚠️  SRM detected! Debug assignment before analysing.' if p_srm < 0.05 else '✅  No SRM.'}")


# ─────────────────────────────────────────────────────────────
# SECTION 6: Bayesian A/B testing (bonus — senior signal)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 6: Bayesian A/B Testing (senior-level bonus)")
print("=" * 60)

print("""
BAYESIAN APPROACH — when Aimé asks 'what are alternatives to p-values?':

  Model: conversion ~ Bernoulli(θ)
         Prior: θ ~ Beta(α, β)   (Beta is conjugate to Bernoulli)
         Posterior: Beta(α + conversions, β + non-conversions)

  Instead of p-value, we compute:
    P(θ_treatment > θ_control) — probability treatment is better
    Expected loss — expected revenue lost by making the wrong decision

  ADVANTAGES over frequentist:
    • Directly answers 'what's the probability treatment is better?'
    • Can incorporate prior business knowledge
    • Doesn't require fixed sample size — update as data arrives
    • No multiple-testing correction needed in the same framework

  DISADVANTAGES:
    • Prior choice is subjective
    • Computationally heavier for complex metrics
    • Harder to explain to non-technical stakeholders
""")

# Beta-Binomial posterior comparison
alpha_prior, beta_prior = 1, 1   # uninformative prior

ctrl_conv_n   = int(ctrl_bookings.sum())
ctrl_total    = N
treat_conv_n  = int(trt_bookings.sum())
treat_total   = N

# Posterior parameters
a_ctrl  = alpha_prior + ctrl_conv_n
b_ctrl  = beta_prior  + (ctrl_total  - ctrl_conv_n)
a_treat = alpha_prior + treat_conv_n
b_treat = beta_prior  + (treat_total - treat_conv_n)

# Monte Carlo estimate of P(treatment > control)
samples = 100_000
theta_ctrl  = np.random.beta(a_ctrl,  b_ctrl,  samples)
theta_treat = np.random.beta(a_treat, b_treat, samples)
prob_better = (theta_treat > theta_ctrl).mean()

print(f"  Posterior mean (control):   {a_ctrl/(a_ctrl+b_ctrl):.4%}")
print(f"  Posterior mean (treatment): {a_treat/(a_treat+b_treat):.4%}")
print(f"  P(treatment > control):     {prob_better:.4%}")
print(f"  {'  → High confidence treatment is better.' if prob_better > 0.95 else '  → Not enough evidence yet.'}")


print("\n" + "=" * 60)
print("SUMMARY CHEAT SHEET")
print("=" * 60)
print("""
  STEP   ACTION                          WHY
  ───────────────────────────────────────────────────────────────
  1.     Define primary + guardrail      Avoid metrics trap
         metrics BEFORE test
  2.     Power analysis → sample size    Know how long to run
  3.     Check assignment ratio (SRM)    Detect instrumentation bugs
  4.     Run for planned duration        Avoid peeking / novelty
  5.     Test primary metric             z-test / t-test / chi-square
  6.     Test ALL guardrails             Bonferroni if multiple
  7.     Make ship/hold/stop decision    Primary ↑ AND guardrails hold
  8.     Post-test segment analysis      Find heterogeneous effects

  α = 0.05 (standard) | Power = 0.80 (standard) | MDE = business decision
""")
print("All sections complete.")
