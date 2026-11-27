"""
The minimal safe clip is over REACHABLE reports, not all reports
================================================================

`experiment.py` computes the critical clip as C* = sup|S| over every report in
(0, 1), and reports:

    Brier        C* = 1.0000
    Spherical    C* = 1.0000
    Logarithmic  C* = 20.7233

That third number is not a property of the logarithmic rule. It is
**-ln(1e-9) = 20.7233**, where 1e-9 is the lower bound of the grid
`critical_clip` happens to scan. Shrink the grid bound and the "constant"
grows without limit: the logarithmic rule has **no finite C\\*** at all.

That is not a bug so much as the wrong question. `C* = sup|S|` is *sufficient*
for propriety -- clip there and the clip never binds -- but it is not *minimal*,
because the supremum can be attained at reports no rational agent would make.
For Brier the supremum sits at q = 1-y, which an agent might plausibly report.
For the logarithmic rule it is approached only as q -> 0 or 1, i.e. by an agent
claiming certainty.

So the real question, and the one worth a theory paper:

    What is sup|S| over the reports a rational agent can actually reach?

This script answers it for a reachable set R(d) = [d, 1-d] -- an agent whose
belief is bounded away from certainty by d. It computes C*(d) analytically and
numerically, measures the DP noise saved (noise scale is 2C*/epsilon, so the
saving is exactly the ratio of clips), and then checks the two things that
decide whether the tighter clip is legitimate:

  SAFETY   does clipping at C*(d) still elicit truthful reports from agents
           whose beliefs lie inside [d, 1-d]?
  COST     what happens to an agent whose belief lies OUTSIDE that range?

Runtime ~20 s, CPU only, no download.
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiment import RULES, brier, logrule, spherical  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))

DELTAS = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]


# ---------------------------------------------------------------------------
# Analytic minimal clips over R(d) = [d, 1-d]
# ---------------------------------------------------------------------------
def analytic_clip(name, d):
    """sup |S(q,y)| for q in [d, 1-d], y in {0,1}.

    Brier      S(q,1) = -(1-q)^2            -> max at q = d      -> (1-d)^2
    Log        S(q,1) = ln q                -> max at q = d      -> -ln d
    Spherical  S(q,1) = q/sqrt(q^2+(1-q)^2) -> max at q = 1-d
    """
    if name == "Brier":
        return (1 - d) ** 2
    if name == "Logarithmic":
        return -np.log(d)
    if name == "Spherical":
        return (1 - d) / np.sqrt((1 - d) ** 2 + d ** 2)
    raise KeyError(name)


def numeric_clip(rule, d, grid=200001):
    qs = np.linspace(d, 1 - d, grid)
    return float(max(np.abs(rule(qs, 1)).max(), np.abs(rule(qs, 0)).max()))


# ---------------------------------------------------------------------------
# Propriety under a clip, restricted to a belief range
# ---------------------------------------------------------------------------
def expected_clipped(rule, q, p, clip):
    s1 = np.clip(rule(q, 1), -clip, clip)
    s0 = np.clip(rule(q, 0), -clip, clip)
    return p * s1 + (1 - p) * s0


def max_distortion(rule, clip, beliefs, report_lo, report_hi, grid=4001):
    """Largest |argmax_q E_p[S_clipped] - p| over the given beliefs."""
    qs = np.linspace(report_lo, report_hi, grid)
    worst = 0.0
    for p in beliefs:
        best = qs[int(np.argmax(expected_clipped(rule, qs, p, clip)))]
        worst = max(worst, abs(best - p))
    return float(worst)


def best_report(rule, clip, p, report_lo, report_hi, grid=4001):
    """Where a clipped rule actually sends an agent holding belief p."""
    qs = np.linspace(report_lo, report_hi, grid)
    return float(qs[int(np.argmax(expected_clipped(rule, qs, p, clip)))])


def main():
    print("=" * 78)
    print("The minimal safe clip is over REACHABLE reports, not all reports")
    print("=" * 78)

    # ---------------- 1. the 20.7233 is a grid artifact -----------------
    print("\n[1/4] Is the logarithmic rule's C* a constant? No -- it is -ln(grid bound).")
    print("      {:>14} {:>14} {:>14}".format("grid lower q", "sup|S| found", "-ln(bound)"))
    for b in [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]:
        qs = np.linspace(b, 1 - b, 200001)
        got = float(max(np.abs(logrule(qs, 1)).max(), np.abs(logrule(qs, 0)).max()))
        note = ""
        if got < -np.log(b) - 1e-6:
            note = "  <- floored by the rule's own EPS=1e-12"
        print("      {:>14.0e} {:>14.4f} {:>14.4f}{}".format(
            b, got, -np.log(b), note))
    print("      -> the reported C* = 20.7233 IS -ln(1e-9). The logarithmic")
    print("         rule has NO finite C* over all of (0,1); any number quoted")
    print("         for it is a statement about the smallest report allowed.")
    print("      The last row makes the point twice: past 1e-12 the number stops")
    print("      tracking the grid and sticks at -ln(1e-12) = 27.6310, because")
    print("      logrule() clips its own argument at EPS. So the 'sensitivity")
    print("      of the logarithmic rule' is set by TWO implementation")
    print("      constants and no mathematics.")
    print("      (Brier and Spherical are genuinely bounded, so theirs are real.)")

    # ---------------- 2. C*(d) over reachable reports -------------------
    print("\n[2/4] C*(d) = sup|S| over reachable reports q in [d, 1-d]")
    print("      analytic vs numeric, and the DP noise saved vs the")
    print("      unrestricted clip (noise scale is 2C*/epsilon).")
    table = {}
    for name in ["Brier", "Logarithmic", "Spherical"]:
        rule = RULES[name]
        base = numeric_clip(rule, 1e-9)       # what experiment.py would report
        rows = []
        print(f"\n      --- {name}   (unrestricted clip {base:.4f})")
        print("      {:>8} {:>12} {:>12} {:>10} {:>14}".format(
            "d", "analytic", "numeric", "rel err", "noise saved"))
        for d in DELTAS:
            a = analytic_clip(name, d)
            n = numeric_clip(rule, d)
            rel = abs(a - n) / max(abs(n), 1e-12)
            saved = base / n
            rows.append({"delta": d, "analytic": a, "numeric": n,
                         "rel_err": rel, "noise_factor": saved})
            print("      {:>8} {:>12.4f} {:>12.4f} {:>10.2e} {:>13.2f}x".format(
                d, a, n, rel, saved))
        table[name] = {"unrestricted": base, "rows": rows}

    print("\n      Brier and Spherical gain almost nothing: their suprema sit")
    print("      at interior-ish reports, so restricting the domain barely")
    print("      moves the clip. The LOGARITHMIC rule is the whole story --")
    print("      its sensitivity is driven entirely by extreme reports.")

    # ---------------- 3. safety: is the tighter clip still proper? ------
    print("\n[3/4] SAFETY -- does clipping at C*(d) keep reports truthful for")
    print("      agents whose beliefs lie INSIDE [d, 1-d]?")
    print("      {:>14} {:>8} {:>12} {:>16}".format(
        "rule", "d", "clip C*(d)", "max distortion"))
    safety = []
    for name in ["Brier", "Logarithmic", "Spherical"]:
        rule = RULES[name]
        for d in [0.1, 0.02, 0.005]:
            c = analytic_clip(name, d)
            beliefs = np.linspace(d, 1 - d, 41)
            dist = max_distortion(rule, c, beliefs, d, 1 - d)
            safety.append({"rule": name, "delta": d, "clip": c, "distortion": dist})
            print("      {:>14} {:>8} {:>12.4f} {:>16.6f}".format(
                name, d, c, dist))
    worst_safe = max(s["distortion"] for s in safety)
    print(f"\n      worst distortion anywhere inside the reachable set: "
          f"{worst_safe:.6f}")
    if worst_safe < 1e-3:
        print("      -> the tighter clip is SAFE for the agents it assumes.")
        print("         Reducing the clip from sup|S| over (0,1) to sup|S| over")
        print("         [d,1-d] costs nothing in incentives, and buys the noise")
        print("         reduction in the table above FOR FREE.")
    else:
        print("      -> the tighter clip DISTORTS even inside the reachable set;")
        print("         C*(d) is not sufficient and the argument fails.")

    # ---------------- 4. cost: agents outside the assumed range --------
    print("\n[4/4] COST -- an agent whose belief is OUTSIDE [d, 1-d]")
    print("      This is what you trade away. Beliefs below d, scored under a")
    print("      clip chosen for the reachable set:")
    print("      {:>14} {:>8} {:>10} {:>12} {:>12}".format(
        "rule", "d", "belief p", "reports q*", "distortion"))
    cost = []
    LO = 1e-4
    for name in ["Brier", "Logarithmic", "Spherical"]:
        rule = RULES[name]
        d = 0.02
        c = analytic_clip(name, d)
        for p in [0.001, 0.005, 0.01]:
            q_star = best_report(rule, c, p, LO, 1 - LO)
            dist = abs(q_star - p)
            cost.append({"rule": name, "delta": d, "belief": p,
                         "best_report": q_star, "distortion": dist})
            print("      {:>14} {:>8} {:>10} {:>12.5f} {:>12.4f}".format(
                name, d, p, q_star, dist))
    at_floor = sum(1 for r in cost if abs(r["best_report"] - LO) < 1e-9)
    print(f"\n      {at_floor}/{len(cost)} of these agents are driven to the very")
    print("      edge of the report space. Read the q* column, not the")
    print("      distortion column: the number looks small only because p is")
    print("      small. What actually happens is that an agent holding p = 0.01")
    print("      is told to claim CERTAINTY. The clip has flattened the part of")
    print("      the score that would have pulled the report back up, so the")
    print("      incentive to be honest about a small probability is gone.")
    print("      That is a qualitative failure, not a 0.0099 one.")
    print("\n      So the trilemma has a third dial nobody turns: you can buy")
    print("      lower DP noise by ASSUMING agents are not near-certain, and")
    print("      pay for it only in the incentives of agents who are.")
    print("      For the logarithmic rule that trade is extremely favourable --")
    print("      the clip grows only as -ln(d), so a 10x tighter assumption")
    print("      costs ~2.3 extra units of clip rather than a factor of 10.")

    # ---------------- figure -------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in ["Brier", "Logarithmic", "Spherical"]:
        ds = [r["delta"] for r in table[name]["rows"]]
        cs = [r["numeric"] for r in table[name]["rows"]]
        ax[0].plot(ds, cs, "o-", label=name)
    ax[0].set_xscale("log")
    ax[0].invert_xaxis()
    ax[0].set_xlabel("d  (reports restricted to [d, 1-d])")
    ax[0].set_ylabel("minimal safe clip  C*(d)")
    ax[0].set_title("C*(d): bounded rules flat, log grows as -ln(d)")
    ax[0].legend(fontsize=8)

    bs = [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
    got = []
    for b in bs:
        qs = np.linspace(b, 1 - b, 50001)
        got.append(float(np.abs(logrule(qs, 1)).max()))
    ax[1].plot(bs, got, "s-", color="crimson", label="sup|S| found")
    ax[1].plot(bs, [-np.log(b) for b in bs], "--", color="black",
               label="-ln(grid bound)")
    ax[1].axhline(20.7233, color="tab:blue", lw=1,
                  label="the 'constant' 20.7233")
    ax[1].set_xscale("log")
    ax[1].invert_xaxis()
    ax[1].set_xlabel("grid lower bound")
    ax[1].set_ylabel("sup |S| for the log rule")
    ax[1].set_title("The log rule's C* is an artifact of the grid")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure_reachable.png"), dpi=130)

    with open(os.path.join(OUT, "results_reachable.json"), "w") as f:
        json.dump({"clips": table, "safety": safety, "cost": cost,
                   "worst_distortion_inside": worst_safe}, f, indent=2)
    print("\nDone. Wrote results_reachable.json and figure_reachable.png")


if __name__ == "__main__":
    main()
