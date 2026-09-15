"""Appendix D and §4: outcome dictionary, primary contrast, bootstrap, secondary analyses, pilot gate.

Scored only from structured actions and the locked payoff table; no LLM judge.
The trajectory (run) is the independent unit; episodes are never treated as independent.
"""
from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Tuple

from . import config as C
from .instrument import ISSUES, OUTSIDE_OPTION, Instrument

SURPLUS_MIN, SURPLUS_MAX = -10.0, 50.0  # bounds for missing-episode sensitivity (main table)


# ---------------------------------------------------------------- episode table
def episode_table(episodes: Iterable[Dict]) -> List[Dict]:
    rows = []
    for ep in episodes:
        inst = Instrument(ep.get("instrument", "main"))
        missing = ep["outcome"] == "TECHNICAL_MISSING"
        agree = None if missing else int(ep["outcome"] == "AGREEMENT")
        pkg = ep.get("accepted_package")
        bu, su = (None, None) if missing else inst.utilities(pkg if agree else None)
        surplus = None if missing else inst.surplus(pkg if agree else None)
        turns = ep.get("turns", [])
        offers = ep.get("offers", [])
        accepted_package = pkg if (agree and not missing) else None
        row = {
            "run_id": ep["run_id"], "component": ep["component"], "pair": ep["pair"], "regime": ep["regime"],
            "block": ep["block"], "episode": ep["episode"], "instrument": inst.name,
            "buyer_model": ep["buyer_model"], "supplier_model": ep["supplier_model"],
            "technical_missing": int(missing), "agreement": agree,
            "accepted_package": accepted_package,
            "first_offer_price": offers[0]["package"]["price"] if offers else None,
            "final_accepted_price": accepted_package["price"] if accepted_package else None,
            "buyer_utility": bu, "supplier_utility": su, "joint_surplus": surplus,
            "ir_agreement": None if missing else int(bool(agree) and bu >= OUTSIDE_OPTION and su >= OUTSIDE_OPTION),
            "violation_buyer": None if missing else int(bool(agree) and bu < OUTSIDE_OPTION),
            "violation_supplier": None if missing else int(bool(agree) and su < OUTSIDE_OPTION),
            "pareto_efficient": (int(inst.is_pareto(pkg)) if agree else None),
            "buyer_share_positive_surplus": _share(bu, su) if agree else None,
            "turns_used": len(turns), "reached_cap": int(ep.get("end_reason") == "CAP"),
            "invalid_buyer": sum(1 for t in turns if t["role"] == "BUYER" and not t["valid"]),
            "invalid_supplier": sum(1 for t in turns if t["role"] == "SUPPLIER" and not t["valid"]),
            "n_turns_buyer": sum(1 for t in turns if t["role"] == "BUYER"),
            "n_turns_supplier": sum(1 for t in turns if t["role"] == "SUPPLIER"),
        }
        row.update(concession_measures(ep.get("offers", []), inst))
        rows.append(row)
    add_stability(rows)
    return rows


def _share(bu, su):
    gb, gs = bu - OUTSIDE_OPTION, su - OUTSIDE_OPTION
    return gb / (gb + gs) if gb > 0 and gs > 0 else None


def concession_measures(offers: List[Dict], inst: Instrument) -> Dict:
    """Concession = previous own-offer utility minus current (positive = concession, negative = retraction).
    Cross-issue change counted separately from joint-value change. Missing if fewer than two own offers."""
    out = {}
    for role, key in (("BUYER", "buyer"), ("SUPPLIER", "supplier")):
        own = [o for o in offers if o["role"] == role]
        if len(own) < 2:
            out.update({key + "_mean_concession": None, key + "_total_concession": None,
                        key + "_retractions": None, key + "_mean_issues_changed": None,
                        key + "_joint_value_change": None})
            continue
        u = [inst.utility(o["package"], role) for o in own]
        conc = [u[i - 1] - u[i] for i in range(1, len(u))]
        changed = [sum(own[i]["package"][k] != own[i - 1]["package"][k] for k in ISSUES) for i in range(1, len(own))]
        joint = [sum(inst.utilities(o["package"])) for o in own]
        out.update({key + "_mean_concession": sum(conc) / len(conc), key + "_total_concession": sum(conc),
                    key + "_retractions": sum(1 for c in conc if c < 0),
                    key + "_mean_issues_changed": sum(changed) / len(changed),
                    key + "_joint_value_change": joint[-1] - joint[0]})
    return out


def add_stability(rows: List[Dict]) -> None:
    """Per-run episode-to-episode fields, including prior-package repeat (App D secondary; edit-plan P3
    item 13 / supervisor-notes DEF-2: repetition/lock-in of the prior accepted package vs renewed search,
    reframed from "collusion" — bilateral bargaining has no excluded third party)."""
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    for rs in by_run.values():
        rs.sort(key=lambda r: r["episode"])
        prev = None
        for r in rs:
            r["stability_abs_change"] = (abs(r["joint_surplus"] - prev["joint_surplus"])
                                         if prev and r["joint_surplus"] is not None and prev["joint_surplus"] is not None
                                         else None)
            r["agreement_transition"] = (f"{prev['agreement']}->{r['agreement']}" if prev else None)
            r["prior_package_repeat"] = (int(prev["accepted_package"] == r["accepted_package"])
                                         if prev and prev["accepted_package"] is not None
                                         and r["accepted_package"] is not None else None)
            prev = r


# ---------------------------------------------------------------- run summaries
def run_summaries(rows: List[Dict], first_episode: int = 2) -> List[Dict]:
    """One row per run. Repeated regimes: mean surplus over episodes 2-4; S: its single episode.
    Includes -10/50 bounds when a scheduled episode is unobserved."""
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    out = []
    for rid, rs in by_run.items():
        r0 = rs[0]
        n_eps = C.EPISODES[r0["regime"]]
        wanted = [1] if n_eps == 1 else list(range(first_episode, n_eps + 1))
        obs = {r["episode"]: r["joint_surplus"] for r in rs if not r["technical_missing"]}
        vals = [obs.get(e) for e in wanted]
        complete = all(v is not None for v in vals)
        lo = [v if v is not None else SURPLUS_MIN for v in vals]
        hi = [v if v is not None else SURPLUS_MAX for v in vals]
        out.append({"run_id": rid, "component": r0["component"], "pair": r0["pair"], "regime": r0["regime"],
                    "block": r0["block"], "complete": complete,
                    "mean_surplus": sum(vals) / len(vals) if complete else None,
                    "bound_low": sum(lo) / len(lo), "bound_high": sum(hi) / len(hi),
                    "agreement_rate": _mean([r["agreement"] for r in rs if r["episode"] in wanted])})
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# ---------------------------------------------------------------- contrasts
def _cell_means(summ: List[Dict], value="mean_surplus") -> Dict[Tuple[str, int, str], float]:
    acc = defaultdict(list)
    for s in summ:
        if s[value] is not None:
            acc[(s["pair"], s["block"], s["regime"])].append(s[value])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def contrast(summ: List[Dict], plus: List[str], minus: List[str], value="mean_surplus") -> Optional[float]:
    """Equal-weight average over pair x block strata of (sum of plus cells) - (sum of minus cells).
    D1-D0: plus=[D1], minus=[D0]. Interaction (D1-D0)-(U1-U0): plus=[D1,U0], minus=[D0,U1]."""
    cm = _cell_means(summ, value)
    strata = {(p, b) for (p, b, _) in cm}
    # equal block weights within pair, then equal pair weights
    by_pair = defaultdict(list)
    for p, b in sorted(strata):
        if all((p, b, g) in cm for g in plus + minus):
            by_pair[p].append(sum(cm[(p, b, g)] for g in plus) - sum(cm[(p, b, g)] for g in minus))
    if not by_pair:
        return None
    return sum(sum(v) / len(v) for v in by_pair.values()) / len(by_pair)


def pair_contrasts(summ, plus, minus, value="mean_surplus") -> Dict[str, Optional[float]]:
    return {p: contrast([s for s in summ if s["pair"] == p], plus, minus, value)
            for p in sorted({s["pair"] for s in summ})}


def bootstrap(summ: List[Dict], plus, minus, value="mean_surplus", reps=C.BOOTSTRAP_RESAMPLES,
              seed=C.BOOTSTRAP_SEED, alpha=0.05) -> Dict:
    """Percentile stratified bootstrap of whole runs within pair x regime x block (named "percentile" per
    supervisor-notes G-07). Returns the resample draws (`draws`) so callers can derive two-sided bootstrap
    p-values without re-resampling (edit-plan P3 item 14)."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for s in summ:
        if s["regime"] in plus + minus and s[value] is not None:
            strata[(s["pair"], s["regime"], s["block"])].append(s)
    est = contrast(summ, plus, minus, value)
    draws = []
    for _ in range(reps):
        sample = []
        for rs in strata.values():
            sample.extend(rng.choice(rs) for _ in rs)
        d = contrast(sample, plus, minus, value)
        if d is not None:
            draws.append(d)
    draws.sort()
    if not draws:
        return {"estimate": est, "ci_low": None, "ci_high": None, "reps": 0, "draws": []}
    lo = draws[int(math.floor(alpha / 2 * len(draws)))]
    hi = draws[min(len(draws) - 1, int(math.ceil((1 - alpha / 2) * len(draws))) - 1)]
    return {"estimate": est, "ci_low": lo, "ci_high": hi, "reps": len(draws), "draws": draws,
            "n_runs": {g: sum(1 for s in summ if s["regime"] == g and s[value] is not None) for g in plus + minus}}


def missing_bounds(summ, plus, minus) -> Dict:
    """Worst/best-case bounds: fill unobserved episodes with -10 or 50 in the direction that moves the contrast."""
    lo_rows, hi_rows = [], []
    for s in summ:
        up = s["regime"] in plus
        lo_rows.append(dict(s, v=s["bound_low"] if up else s["bound_high"]))
        hi_rows.append(dict(s, v=s["bound_high"] if up else s["bound_low"]))
    return {"low": contrast(lo_rows, plus, minus, "v"), "high": contrast(hi_rows, plus, minus, "v"),
            "incomplete_runs": {g: sum(1 for s in summ if s["regime"] == g and not s["complete"]) for g in plus + minus}}


def standard_error_multiplier(n_per_cell: int = C.MAIN_RUNS_PER_CELL) -> float:
    """SE of a difference of two cell means in SD units: sqrt(2/n). n=24 -> 0.289 (§4.2)."""
    return math.sqrt(2.0 / n_per_cell)


def minimum_detectable_effect(sd: float, n_per_cell: int = C.MAIN_RUNS_PER_CELL, z_alpha=1.96, z_power=0.84) -> float:
    """Pilot SD -> numeric MDE in utility points (Appendix C)."""
    return (z_alpha + z_power) * sd * standard_error_multiplier(n_per_cell)


def mde_sd_units(n: int = C.MAIN_RUNS_PER_CELL, power: float = 0.8, alpha: float = 0.05) -> float:
    """A-priori minimum detectable effect in trajectory-SD units, independent of any pilot SD estimate:
    (z_{1-alpha/2} + z_power) * sqrt(2/n). n=24, power=.8, alpha=.05 -> ~0.81 (Appendix C/D; edit-plan
    P3 item 14; supervisor-notes ABC-6/G-02: 80% power, alpha=.05 -> MDE ~0.81 SD)."""
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    return (z_alpha + z_power) * standard_error_multiplier(n)


# ---------------------------------------------------------------- rank / interval statistics (stdlib only)
def _ranks(xs: Tuple[float, ...]) -> List[float]:
    """1-based ranks with ties resolved to the average rank of the tied block."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs: Iterable[Optional[float]], ys: Iterable[Optional[float]]) -> Optional[float]:
    """Spearman rank correlation, stdlib only (Takenami et al. 2025 Findings-EMNLP first-offer anchoring;
    edit-plan P3 item 13). Pairs with a missing value are dropped; None if fewer than two pairs remain or
    either series is constant (undefined correlation)."""
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pts) < 2:
        return None
    xs2, ys2 = zip(*pts)
    rx, ry = _ranks(xs2), _ranks(ys2)
    if len(set(rx)) == 1 or len(set(ry)) == 1:
        return None
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if sx == 0 or sy == 0:
        return None
    return cov / (sx * sy)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[Optional[float], Optional[float]]:
    """Wilson score 95% interval for a binomial rate (default z = 1.96 for 95%); for cell-level agreement
    and IR-agreement rates, one-shot and per-episode (edit-plan P3 item 14; supervisor-notes G-06).
    Trajectory-level rates keep the run-level stratified bootstrap."""
    if n == 0:
        return None, None
    phat = successes / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom)


def bootstrap_two_sided_p(draws: List[float]) -> Optional[float]:
    """Two-sided bootstrap p-value from percentile-bootstrap draws of a contrast: 2 * min(share <= 0,
    share >= 0), capped at 1 (Appendix D; edit-plan P3 item 14; supervisor-notes G-04)."""
    if not draws:
        return None
    n = len(draws)
    share_le0 = sum(1 for d in draws if d <= 0) / n
    share_ge0 = sum(1 for d in draws if d >= 0) / n
    return min(1.0, 2 * min(share_le0, share_ge0))


def benjamini_hochberg(pvalues):
    """Benjamini-Hochberg q-values. Accepts a list (returns a list in the same order) or a dict (returns a
    dict with the same keys); q-values are monotone non-decreasing from the largest p-value down (Appendix
    D; edit-plan P3 item 14; supervisor-notes G-04: exploratory secondaries + BH q-values)."""
    if isinstance(pvalues, dict):
        names = list(pvalues.keys())
        qs = benjamini_hochberg([pvalues[n] for n in names])
        return dict(zip(names, qs))
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    raw = [pvalues[order[i]] * m / (i + 1) for i in range(m)]
    q_sorted = [0.0] * m
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        running_min = min(running_min, raw[i])
        q_sorted[i] = min(1.0, running_min)
    out = [0.0] * m
    for i, idx in enumerate(order):
        out[idx] = q_sorted[i]
    return out


def secondary_family_bh(summ: List[Dict], value: str = "mean_surplus") -> Dict[str, Dict]:
    """Exploratory secondary family (edit-plan P3 item 14; supervisor-notes G-04): pair-specific D1-D0
    effects, the interaction contrast, and the D0-S / D1-S descriptive contrasts. Two-sided bootstrap
    p-values per member, then one shared Benjamini-Hochberg pass. Labelled exploratory throughout; never
    used in place of the pre-registered primary contrast."""
    pairs = sorted({s["pair"] for s in summ})
    members = {}
    for p in pairs:
        members["D1_minus_D0_" + p] = bootstrap([s for s in summ if s["pair"] == p], ["D1"], ["D0"], value)
    members["interaction"] = bootstrap(summ, ["D1", "U0"], ["D0", "U1"], value)
    members["D0_minus_S"] = bootstrap(summ, ["D0"], ["S"], value)
    members["D1_minus_S"] = bootstrap(summ, ["D1"], ["S"], value)
    pvals = {name: bootstrap_two_sided_p(bs["draws"]) for name, bs in members.items()}
    qvals = benjamini_hochberg({k: v for k, v in pvals.items() if v is not None})
    return {name: {"estimate": members[name]["estimate"], "p_value": pvals[name], "q_value": qvals.get(name)}
            for name in members}


def prior_package_repeat_by_cell(rows: List[Dict]) -> List[Dict]:
    """Cell-level rate of prior-package repeat (App D secondary; edit-plan P3 item 13)."""
    acc = defaultdict(list)
    for r in rows:
        if r.get("prior_package_repeat") is not None:
            acc[(r["component"], r["pair"], r["regime"], r["episode"])].append(r["prior_package_repeat"])
    return [{"component": c, "pair": p, "regime": g, "episode": e, "n": len(v), "repeat_rate": sum(v) / len(v)}
            for (c, p, g, e), v in sorted(acc.items())]


def anchoring_by_cell(rows: List[Dict]) -> List[Dict]:
    """Cell-level Spearman correlation between the episode's first valid offer price and the final
    accepted price (App D secondary; edit-plan P3 item 13)."""
    acc = defaultdict(list)
    for r in rows:
        if r.get("first_offer_price") is not None:
            acc[(r["component"], r["pair"], r["regime"], r["episode"])].append(r)
    out = []
    for (c, p, g, e), rs in sorted(acc.items()):
        xs = [r["first_offer_price"] for r in rs]
        ys = [r["final_accepted_price"] for r in rs]
        out.append({"component": c, "pair": p, "regime": g, "episode": e, "n": len(rs),
                    "spearman_first_offer_vs_final_price": spearman(xs, ys)})
    return out


def bootstrap_coverage(sd: float = 10.0, n_per_cell: int = C.MAIN_RUNS_PER_CELL, true_effect: float = 5.0,
                       sims: int = 200, reps: int = 200, seed: int = C.BOOTSTRAP_SEED, alpha: float = 0.05) -> float:
    """Simulate normal trajectory-level outcomes at a fixed effect and SD, run the percentile stratified
    bootstrap on each simulated pilot, and report empirical 95% CI coverage (edit-plan P3 item 14;
    supervisor-notes G-07: name the bootstrap "percentile", simulate coverage at pilot SD before main
    collection). Defaults kept small for speed; not a claim about the real pilot's SD."""
    rng = random.Random(seed)
    blocks = list(range(4))
    per_block = max(1, n_per_cell // len(blocks))
    covered = 0
    for i in range(sims):
        summ = []
        for b in blocks:
            for _ in range(per_block):
                summ.append({"pair": "p0", "block": b, "regime": "D0", "mean_surplus": rng.gauss(0.0, sd)})
                summ.append({"pair": "p0", "block": b, "regime": "D1", "mean_surplus": rng.gauss(true_effect, sd)})
        bs = bootstrap(summ, ["D1"], ["D0"], value="mean_surplus", reps=reps,
                       seed=rng.randrange(1 << 30), alpha=alpha)
        if bs["ci_low"] is not None and bs["ci_low"] <= true_effect <= bs["ci_high"]:
            covered += 1
    return covered / sims if sims else 0.0


# ---------------------------------------------------------------- reports
def secondary_by_cell(rows: List[Dict]) -> List[Dict]:
    """Cell-level descriptives, including Wilson 95% intervals for agreement/IR-agreement rates (edit-plan
    P3 item 14 / G-06), the prior-package repeat rate (item 13 / DEF-2) and the first-offer-anchoring
    Spearman correlation (item 13). Descriptive only; the primary contrast lives in `full_report`."""
    acc = defaultdict(list)
    for r in rows:
        acc[(r["component"], r["pair"], r["regime"], r["episode"])].append(r)
    out = []
    for (comp, pair, regime, ep), rs in sorted(acc.items()):
        obs = [r for r in rs if not r["technical_missing"]]
        agr = [r for r in obs if r["agreement"]]
        n_obs = len(obs)
        agr_lo, agr_hi = wilson_interval(sum(r["agreement"] for r in obs), n_obs) if n_obs else (None, None)
        ir_lo, ir_hi = wilson_interval(sum(r["ir_agreement"] for r in obs), n_obs) if n_obs else (None, None)
        repeats = [r["prior_package_repeat"] for r in obs if r.get("prior_package_repeat") is not None]
        anchor_pairs = [(r["first_offer_price"], r["final_accepted_price"]) for r in obs
                        if r.get("first_offer_price") is not None]
        ax, ay = zip(*anchor_pairs) if anchor_pairs else ((), ())
        out.append({"component": comp, "pair": pair, "regime": regime, "episode": ep, "n": len(rs),
                    "technical_missing": len(rs) - len(obs),
                    "agreement_rate": _mean([r["agreement"] for r in obs]),
                    "agreement_wilson_lo": agr_lo, "agreement_wilson_hi": agr_hi,
                    "ir_agreement_rate": _mean([r["ir_agreement"] for r in obs]),
                    "ir_agreement_wilson_lo": ir_lo, "ir_agreement_wilson_hi": ir_hi,
                    "mean_surplus": _mean([r["joint_surplus"] for r in obs]),
                    "mean_buyer_utility": _mean([r["buyer_utility"] for r in obs]),
                    "mean_supplier_utility": _mean([r["supplier_utility"] for r in obs]),
                    "violations": sum(r["violation_buyer"] + r["violation_supplier"] for r in obs),
                    "pareto_rate_among_agreements": _mean([r["pareto_efficient"] for r in agr]),
                    "mean_buyer_share": _mean([r["buyer_share_positive_surplus"] for r in agr]),
                    "mean_turns": _mean([r["turns_used"] for r in obs]),
                    "cap_rate": _mean([r["reached_cap"] for r in obs]),
                    "mean_buyer_concession": _mean([r["buyer_mean_concession"] for r in obs]),
                    "mean_supplier_concession": _mean([r["supplier_mean_concession"] for r in obs]),
                    "mean_stability_abs_change": _mean([r["stability_abs_change"] for r in obs]),
                    "prior_package_repeat_rate": sum(repeats) / len(repeats) if repeats else None,
                    "first_offer_anchoring_spearman": spearman(ax, ay)})
    return out


def pilot_gate(rows: List[Dict]) -> Dict:
    """Appendix C interface thresholds. Never used to pick cooperative models or prompts."""
    per_model = defaultdict(lambda: [0, 0, 0, 0])  # invalid, turns, cap episodes, episodes
    for r in rows:
        for role in ("buyer", "supplier"):
            m = r[role + "_model"]
            per_model[m][0] += r["invalid_" + role]
            per_model[m][1] += r["n_turns_" + role]
            per_model[m][2] += r["reached_cap"]
            per_model[m][3] += 1
    tot_inv = sum(v[0] for v in per_model.values())
    tot_turns = sum(v[1] for v in per_model.values())
    overall = tot_inv / tot_turns if tot_turns else 0.0
    models = {m: {"invalid_rate": v[0] / v[1] if v[1] else 0.0, "cap_rate": v[2] / v[3] if v[3] else 0.0}
              for m, v in per_model.items()}
    flags = []
    if overall > C.PILOT_MAX_INVALID_OVERALL:
        flags.append("overall invalid rate {:.1%} > 5%".format(overall))
    for m, v in models.items():
        if v["invalid_rate"] > C.PILOT_MAX_INVALID_PER_MODEL:
            flags.append("{} invalid rate {:.1%} > 10%".format(m, v["invalid_rate"]))
        if v["cap_rate"] > C.PILOT_MAX_CAP_REACHED_PER_MODEL:
            flags.append("{} cap-reached rate {:.1%} > 25%: review message cap".format(m, v["cap_rate"]))
    sds = [r["joint_surplus"] for r in rows if r["joint_surplus"] is not None]
    sd = _sd(sds)
    return {"overall_invalid_rate": overall, "models": models, "flags": flags, "protocol_review": bool(flags),
            "pilot_surplus_sd": sd, "mde_utility_points": minimum_detectable_effect(sd) if sd else None}


def _sd(xs):
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def illustration_sample(summ: List[Dict], seed=C.ILLUSTRATION_SEED) -> Dict[str, str]:
    """One complete run per main pair-regime cell, drawn with the archived seed."""
    rng = random.Random(seed)
    cells = defaultdict(list)
    for s in summ:
        if s["component"] == "main" and s["complete"]:
            cells[(s["pair"], s["regime"])].append(s["run_id"])
    return {"{}|{}".format(*k): rng.choice(sorted(v)) for k, v in sorted(cells.items())}


def full_report(episodes: List[Dict]) -> Dict:
    """Confirmatory report. Restricted to component in (main, extension): placebo_pilot and paraphrase_pilot
    (and pilot/robustness_pilot/feasibility) never enter this report, even if passed in by mistake — they
    surface only through pilot_gate and the pilot-batch outputs (edit-plan-2026-09-15.md P3 items 10 & 12)."""
    episodes = [e for e in episodes if e["component"] in ("main", "extension")]
    rows = episode_table(episodes)
    main_rows = [r for r in rows if r["component"] == "main"]
    summ = run_summaries(main_rows)
    ext = run_summaries([r for r in rows if r["component"] == "extension"])
    return {
        "counts": {"episodes": len(rows), "main_runs": len(summ), "extension_runs": len(ext)},
        "primary_D1_minus_D0": bootstrap(summ, ["D1"], ["D0"]),
        "primary_bounds": missing_bounds(summ, ["D1"], ["D0"]),
        "primary_by_pair": pair_contrasts(summ, ["D1"], ["D0"]),
        "interaction_(D1-D0)-(U1-U0)": bootstrap(summ, ["D1", "U0"], ["D0", "U1"]),
        "descriptive_D0_minus_S": bootstrap(summ, ["D0"], ["S"]),
        "descriptive_D1_minus_S": bootstrap(summ, ["D1"], ["S"]),
        "agreement_D1_minus_D0": contrast(summ, ["D1"], ["D0"], "agreement_rate"),
        "extension_D1_descriptive": {s["pair"]: _mean([x["mean_surplus"] for x in ext if x["pair"] == s["pair"]])
                                     for s in ext},
        "se_multiplier_n24": standard_error_multiplier(),
        "mde_a_priori_sd_units": mde_sd_units(),
        "secondary_family_bh": secondary_family_bh(summ),
        "prior_package_repeat_by_cell": prior_package_repeat_by_cell(main_rows),
        "first_offer_anchoring_by_cell": anchoring_by_cell(main_rows),
        "illustrations": illustration_sample(summ),
        "caveat": "Configuration effects for archived models and one synthetic case; exploratory secondaries.",
    }


def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
