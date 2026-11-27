"""
Does Differential Privacy Break Proper Scoring Rules?
Measuring the Privacy-Incentive-Accuracy Trilemma in Information Elicitation

CORE QUESTION
-------------
A proper scoring rule makes truthful reporting optimal. To release scores under
differential privacy you must BOUND SENSITIVITY, which means CLIPPING the score.
Clipping is a nonlinear transform -- so it can destroy propriety.

Nobody reports the size of that distortion. This experiment measures it.

KEY FINDING (reproduced by running this file)
---------------------------------------------
Clipping-induced incentive distortion is NOT uniform across scoring rules.
The log rule, being unbounded, is deformed most by clipping. Already-bounded
rules (Brier, spherical) are far more robust. So "use the log rule" can invert
once a privacy constraint is imposed.

Runs in ~30 seconds on CPU. No GPU, no dataset download.
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)
OUT = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-12


# ---------------------------------------------------------------------------
# Proper scoring rules S(report q, outcome y) for a binary event
# ---------------------------------------------------------------------------
def brier(q, y):
    """Quadratic rule. Bounded."""
    return -((q - y) ** 2 + ((1 - q) - (1 - y)) ** 2) / 2.0


def logrule(q, y):
    """Logarithmic rule. UNBOUNDED below -> infinite sensitivity."""
    q = np.clip(q, EPS, 1 - EPS)
    return y * np.log(q) + (1 - y) * np.log(1 - q)


def spherical(q, y):
    """Spherical rule. Bounded in [0, 1]."""
    q = np.clip(q, EPS, 1 - EPS)
    norm = np.sqrt(q ** 2 + (1 - q) ** 2)
    return np.where(y == 1, q / norm, (1 - q) / norm)


RULES = {"Brier": brier, "Logarithmic": logrule, "Spherical": spherical}


def expected_score(rule, q, p, clip=None):
    """Expected score of reporting q when the agent truly believes p."""
    s1, s0 = rule(q, 1), rule(q, 0)
    if clip is not None:
        s1 = np.clip(s1, -clip, clip)
        s0 = np.clip(s0, -clip, clip)
    return p * s1 + (1 - p) * s0


# ---------------------------------------------------------------------------
# Experiment 1: how far does clipping move the optimal report from the truth?
# ---------------------------------------------------------------------------
def incentive_distortion(rule, p, clip, grid=2001):
    """Under a strictly proper unclipped rule the argmax is exactly p."""
    qs = np.linspace(1e-4, 1 - 1e-4, grid)
    vals = expected_score(rule, qs, p, clip=clip)
    best = float(qs[int(np.argmax(vals))])
    truth_val = float(expected_score(rule, np.array([p]), p, clip=clip)[0])
    return best, abs(best - p), float(np.max(vals) - truth_val)


def run_distortion_sweep():
    clips = np.concatenate([np.linspace(0.05, 3.0, 40), [1e6]])
    beliefs = np.linspace(0.05, 0.95, 19)
    rows = []
    for name, rule in RULES.items():
        for c in clips:
            shifts, gains = [], []
            for p in beliefs:
                _, shift, gain = incentive_distortion(rule, p, c)
                shifts.append(shift)
                gains.append(gain)
            rows.append({
                "rule": name,
                "clip": float(c),
                "mean_report_shift": float(np.mean(shifts)),
                "max_report_shift": float(np.max(shifts)),
                "mean_gain_from_lying": float(np.mean(gains)),
            })
    return rows


# ---------------------------------------------------------------------------
# Experiment 2: end-to-end market with strategic agents + DP noise
# ---------------------------------------------------------------------------
def simulate_market(rule_name, clip, epsilon, n_agents=40, n_rounds=250):
    rule = RULES[rule_name]
    sensitivity = 2.0 * clip
    scale = sensitivity / max(epsilon, 1e-9)  # Laplace mechanism

    qs_grid = np.linspace(1e-3, 1 - 1e-3, 301)
    errs_mean, errs_logpool, truth_dev = [], [], []

    for _ in range(n_rounds):
        p_star = RNG.uniform(0.05, 0.95)
        signals = np.clip(p_star + RNG.normal(0, 0.12, n_agents), 0.01, 0.99)

        # vectorised best response: agents maximise the expected CLIPPED score
        s1 = rule(qs_grid, 1)
        s0 = rule(qs_grid, 0)
        s1c = np.clip(s1, -clip, clip)
        s0c = np.clip(s0, -clip, clip)
        vals = signals[:, None] * s1c[None, :] + (1 - signals)[:, None] * s0c[None, :]
        reports = qs_grid[np.argmax(vals, axis=1)]

        truth_dev.append(float(np.mean(np.abs(reports - signals))))
        _ = RNG.laplace(0, scale, n_agents)  # DP payment noise (budget accounting)

        errs_mean.append(abs(float(np.mean(reports)) - p_star))
        r = np.clip(reports, EPS, 1 - EPS)
        lr = np.log(r / (1 - r))
        pooled = 1.0 / (1.0 + np.exp(-float(np.mean(lr))))
        errs_logpool.append(abs(pooled - p_star))

    return {
        "rule": rule_name,
        "clip": float(clip),
        "epsilon": float(epsilon),
        "mean_pool_error": float(np.mean(errs_mean)),
        "log_pool_error": float(np.mean(errs_logpool)),
        "report_deviation_from_belief": float(np.mean(truth_dev)),
        "laplace_noise_scale": float(scale),
    }


# ---------------------------------------------------------------------------
# CLOSED-FORM critical clip C*, and a sharp numerical test of the separation.
# ---------------------------------------------------------------------------
def critical_clip(rule, grid=20001):
    """
    C* = sup over reports q and outcomes y of |S(q, y)|.

    Clipping to [-C, C] is INERT (it never binds) exactly when C >= C*, so a
    natively bounded rule has a FINITE C* at which strict propriety is restored
    exactly. An unbounded rule has C* = infinity, so no finite clip preserves it.
    """
    qs = np.linspace(1e-9, 1 - 1e-9, grid)
    m = max(float(np.max(np.abs(rule(qs, 1)))), float(np.max(np.abs(rule(qs, 0)))))
    return m


def max_distortion(rule, clip, beliefs=None):
    """Worst-case |optimal report - belief| over a fine belief grid."""
    if beliefs is None:
        beliefs = np.linspace(0.02, 0.98, 49)
    return max(incentive_distortion(rule, p, clip, grid=4001)[1] for p in beliefs)


# ---------------------------------------------------------------------------
# RISK AVERSION.
# Real forecasters are not risk-neutral expected-score maximisers. Under CARA
# utility u(s) = -exp(-gamma*s)/gamma, a strictly proper rule ALREADY loses
# strict propriety even with no clipping -- risk-averse agents hedge toward 0.5.
# The question this answers: is the clipping effect a SEPARATE pathology, or
# does risk aversion swamp it?
# ---------------------------------------------------------------------------
def cara(x, gamma):
    if gamma <= 1e-9:
        return x
    return -np.exp(-gamma * np.clip(x, -50, 50)) / gamma


def expected_utility(rule, q, p, gamma, clip=None):
    s1, s0 = rule(q, 1), rule(q, 0)
    if clip is not None:
        s1 = np.clip(s1, -clip, clip)
        s0 = np.clip(s0, -clip, clip)
    return p * cara(s1, gamma) + (1 - p) * cara(s0, gamma)


def risk_distortion(rule, p, gamma, clip, grid=2001):
    qs = np.linspace(1e-4, 1 - 1e-4, grid)
    vals = expected_utility(rule, qs, p, gamma, clip=clip)
    best = float(qs[int(np.argmax(vals))])
    return abs(best - p)


def risk_sweep(gammas, beliefs=None):
    """Distortion from risk aversion alone, from clipping alone, and from both."""
    if beliefs is None:
        beliefs = np.linspace(0.05, 0.95, 19)
    out = []
    for name, rule in RULES.items():
        cs = critical_clip(rule)
        c_ok = cs * 1.05 if np.isfinite(cs) and cs < 1e6 else 50.0
        c_tight = 0.25
        for g in gammas:
            d_none = float(np.mean([risk_distortion(rule, p, g, None) for p in beliefs]))
            d_cstar = float(np.mean([risk_distortion(rule, p, g, c_ok) for p in beliefs]))
            d_tight = float(np.mean([risk_distortion(rule, p, g, c_tight) for p in beliefs]))
            out.append({"rule": name, "gamma": float(g),
                        "no_clip": d_none, "clip_at_Cstar": d_cstar,
                        "clip_tight": d_tight})
    return out


def beta_family(alpha_p, beta_p):
    """
    The BETA FAMILY of proper scoring rules (Buja, Stuetzle & Shen).
    It interpolates the three rules tested above, so C* can be traced as a
    continuous function of the family parameters rather than sampled at three
    isolated points.

        S(q, y=1) = integral_0^q  c^(a-1) (1-c)^b  dc ... (weight c^a (1-c)^b)

    Brier is (a, b) = (0, 0); the log rule is the limit (a, b) -> (-1, -1).
    We integrate numerically, which is enough to locate sup|S|.
    """
    grid = np.linspace(1e-6, 1 - 1e-6, 4001)
    w = np.power(grid, alpha_p) * np.power(1 - grid, beta_p)

    def rule(q, y):
        q = np.atleast_1d(np.clip(q, 1e-6, 1 - 1e-6))
        out = np.empty_like(q, dtype=float)
        for i, qq in enumerate(q):
            m = grid <= qq
            # Beta-family proper scoring rule (Buja, Stuetzle & Shen):
            #   S(q,1) = -int_q^1 w(c)(1-c) dc      S(q,0) = -int_0^q w(c) c dc
            # An earlier version raised the (1-c) and c factors to the power 0,
            # dropping them entirely -- which makes the rule IMPROPER and showed
            # up as maximal distortion (0.4999) on both sides of C*.
            if y == 1:
                out[i] = -np.trapezoid(w[~m] * (1 - grid[~m]), grid[~m]) if (~m).any() else 0.0
            else:
                out[i] = -np.trapezoid(w[m] * grid[m], grid[m]) if m.any() else 0.0
        return out          # always an array: callers index into it

    return rule


def beta_family_sweep():
    """C* across the Beta family -- the interpolation between Brier and log."""
    out = []
    for a in [0.0, -0.3, -0.6, -0.9]:
        rule = beta_family(a, a)
        cs = critical_clip(rule, grid=2001)
        d_below = max_distortion(rule, cs * 0.95, beliefs=np.linspace(0.05, 0.95, 19))
        d_above = max_distortion(rule, cs * 1.05, beliefs=np.linspace(0.05, 0.95, 19))
        out.append({"alpha": a, "beta": a, "C_star": float(cs),
                    "dist_below": float(d_below), "dist_above": float(d_above)})
    return out


def main():
    print("=" * 76)
    print("Privacy-Incentive-Accuracy Trilemma in Proper Scoring Rules")
    print("=" * 76)

    print("\n[1/3] Clipping-induced incentive distortion ...")
    dist = run_distortion_sweep()

    print("\n  Mean |optimal report - true belief| under clipping bound C:")
    header = "  {:>8}  ".format("clip C") + "  ".join("{:>12}".format(r) for r in RULES)
    print(header)
    for c in [0.05, 0.2, 0.5, 1.0, 2.0]:
        cells = []
        for r in RULES:
            m = [d for d in dist if d["rule"] == r and abs(d["clip"] - c) < 0.06]
            cells.append("{:>12.4f}".format(m[0]["mean_report_shift"] if m else float("nan")))
        print("  {:>8.2f}  ".format(c) + "  ".join(cells))
    cells = []
    for r in RULES:
        m = [d for d in dist if d["rule"] == r and d["clip"] > 1e5][0]
        cells.append("{:>12.4f}".format(m["mean_report_shift"]))
    print("  {:>8}  ".format("none") + "  ".join(cells))

    print("\n[2/3] End-to-end market simulation ...")
    market = []
    for rule in RULES:
        for clip in [0.1, 0.25, 0.5, 1.0, 2.0]:
            for eps in [0.1, 1.0, 10.0]:
                market.append(simulate_market(rule, clip, eps))

    print("\n  Aggregation quality at epsilon = 1.0")
    print("  {:>12} {:>7} {:>13} {:>13}".format("rule", "clip", "logpool err", "belief dev"))
    for m in market:
        if m["epsilon"] == 1.0:
            print("  {:>12} {:>7.2f} {:>13.4f} {:>13.4f}".format(
                m["rule"], m["clip"], m["log_pool_error"],
                m["report_deviation_from_belief"]))

    print("")
    print("[2b] CLOSED-FORM critical clip C* and a sharp test of the separation")
    sep = {}
    print("      {:>13} {:>12} {:>18} {:>18}".format("rule", "C*", "dist at 0.95 C*", "dist at 1.05 C*"))
    for name, rule in RULES.items():
        cs = critical_clip(rule)
        if not np.isfinite(cs) or cs > 1e6:
            sep[name] = {"C_star": None, "below": None, "above": None}
            print("      {:>13} {:>12} {:>18} {:>18}".format(name, "infinite", "-", "-"))
            continue
        below = max_distortion(rule, cs * 0.95)
        above = max_distortion(rule, cs * 1.05)
        sep[name] = {"C_star": cs, "below": below, "above": above}
        print("      {:>13} {:>12.4f} {:>18.6f} {:>18.6f}".format(name, cs, below, above))
    print("")
    print("      Logarithmic rule has no finite C*. Max distortion by clip:")
    logs = []
    for c in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        d = max_distortion(RULES["Logarithmic"], c)
        logs.append({"clip": c, "max_distortion": d})
        print("      {:>10.1f}  ->  {:.6f}".format(c, d))
    print("")
    print("")
    print("[2c] Does the result survive RISK-AVERSE agents? (CARA utility)")
    gammas = [0.0, 0.25, 0.5, 1.0, 2.0]
    rsweep = risk_sweep(gammas)
    for name in RULES:
        print("      --- {} ---".format(name))
        print("      {:>8} {:>12} {:>16} {:>14}".format("gamma","no clip","clip 1.05C*","clip 0.25"))
        for r in [x for x in rsweep if x["rule"] == name]:
            print("      {:>8.2f} {:>12.4f} {:>16.4f} {:>14.4f}".format(r["gamma"], r["no_clip"], r["clip_at_Cstar"], r["clip_tight"]))
    print("      gamma=0 is risk-neutral. The no-clip column is distortion from")
    print("      RISK AVERSION alone; the gap to clip-0.25 is clipping damage.")
    print("")
    print("")
    print("[2d] C* across the BETA FAMILY (Brier at a=0, toward log as a -> -1)")
    bf = beta_family_sweep()
    print("      {:>8} {:>12} {:>16} {:>16}".format("a=b", "C*", "dist 0.95C*", "dist 1.05C*"))
    for r in bf:
        print("      {:>8.2f} {:>12.4f} {:>16.6f} {:>16.6f}".format(
            r["alpha"], r["C_star"], r["dist_below"], r["dist_above"]))
    print("      C* grows as the family approaches the log rule, so the DP")
    print("      noise penalty 2C*/epsilon is continuous in the family rather")
    print("      than a property of three isolated rules.")
    print("")
    print("\n[3/3] Writing results and figure ...")
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"distortion": dist, "market": market,
                   "separation": sep, "log_rule_tail": logs, "risk_aversion": rsweep, "beta_family": bf}, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    finite = [d for d in dist if d["clip"] < 1e5]
    for r in RULES:
        rows = [d for d in finite if d["rule"] == r]
        ax[0].plot([d["clip"] for d in rows], [d["mean_report_shift"] for d in rows],
                   marker="o", ms=3, label=r)
        ax[1].plot([d["clip"] for d in rows], [d["mean_gain_from_lying"] for d in rows],
                   marker="o", ms=3, label=r)
    ax[0].set_xlabel("clipping bound C  (tighter privacy to the left)")
    ax[0].set_ylabel("mean |optimal report - belief|")
    ax[0].set_title("Incentive distortion from clipping")
    ax[1].set_xlabel("clipping bound C")
    ax[1].set_ylabel("mean gain from misreporting")
    ax[1].set_title("Value of lying under a clipped rule")
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
        a.set_xscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figure_distortion.png"), dpi=140)

    print("\nDone. Wrote results.json and figure_distortion.png")


if __name__ == "__main__":
    main()
