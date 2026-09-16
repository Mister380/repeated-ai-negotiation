"""Command line entry point.

  python -m negotiation_eval check-instrument
  python -m negotiation_eval schedule   [--design three_group|fallback] [--out runs/]
  python -m negotiation_eval feasibility --models opus,haiku,gpt55,luna,deepseek,qwen,kimi,glm --budget-usd 5
  python -m negotiation_eval pilot      --budget-usd 60 [--design ...]
  python -m negotiation_eval run        --budget-usd 1200 --batch main-v1 [--design ...] [--limit N]
  python -m negotiation_eval analyze    --batch main-v1 [--pilot]
  python -m negotiation_eval simulate   (deterministic synthetic calibration; no network, no cost)
  python -m negotiation_eval dry-run    (full pipeline on a mock provider, no network, no cost)
"""
from __future__ import annotations

import argparse
import json
import os
import random

from . import analysis as A
from . import config as C
from . import feasibility as F
from . import schedule as S
from .instrument import Instrument
from .providers import MockProvider, default_providers
from .runner import Runner


def mock_policy(seed=7):
    """A naive scripted agent: random valid offers, accepts anything at or above 40 after message 4."""
    rng = random.Random(seed)
    import re
    from .protocol import parse_action  # noqa: F401

    def policy(spec, prompt):
        inst = Instrument("robustness" if "| sla | PREMIUM (99.9% availability and 1-hour response) | 40 |" in prompt
                          or "| sla | BASIC (99.0% availability and 8-hour response) | 10 |" in prompt else "main")
        # Role and remaining-message count are read positionally so this works for any INSTRUCTION_VARIANTS
        # wording (canonical or paraphrase_v1): the instruction paragraph comes first in the prompt and is
        # the only place BUYER/SUPPLIER and the remaining-count appear before the transcript.
        idx_b, idx_s = prompt.find("BUYER"), prompt.find("SUPPLIER")
        role = "BUYER" if idx_b != -1 and (idx_s == -1 or idx_b < idx_s) else "SUPPLIER"
        eid = re.search(r"Episode identifier: (\S+)", prompt).group(1)
        remaining = int(re.search(r"(\d+)\s+(?:messages left|of them left)", prompt).group(1))
        n = C.MAX_MESSAGES - remaining + 1
        other = "SUPPLIER" if role == "BUYER" else "BUYER"
        offers = re.findall(r"\] " + other + r" (\{.*\"action\": \"OFFER\".*\})", prompt)
        if offers and n > 4:
            last = json.loads(offers[-1])
            pkg = {k: last[k] for k in ("price", "sla", "duration_months", "exit_days", "scope")}
            if inst.utility(pkg, role) >= 40 or rng.random() < 0.05:
                return json.dumps({"action": "ACCEPT", "offer_id": last["offer_id"], "message": "Agreed."})
        if rng.random() < 0.03:
            return "I think we should talk about price."  # invalid on purpose
        pkg = rng.choice(inst.all_packages())
        return json.dumps(dict(action="OFFER", offer_id="{}-m{}".format(eid, n), message="Proposal.", **pkg))
    return policy


def main(argv=None):
    ap = argparse.ArgumentParser(prog="negotiation_eval")
    ap.add_argument("command", choices=["check-instrument", "schedule", "feasibility", "pilot", "run", "analyze",
                                        "simulate", "dry-run"])
    ap.add_argument("--design", default="three_group", choices=["three_group", "six_provider", "fallback"])
    ap.add_argument("--out", default="data")
    ap.add_argument("--batch", default=None)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--models", default=",".join(C.MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pilot", action="store_true", help="analyze: apply pilot gate instead of confirmatory report")
    ap.add_argument("--seed", type=int, default=C.BOOTSTRAP_SEED, help="simulate: deterministic synthetic seed")
    ap.add_argument("--reps", type=int, default=500, help="simulate: calibration resamples")
    a = ap.parse_args(argv)

    if a.command == "check-instrument":
        print(json.dumps({n: Instrument(n).enumerate_properties() for n in ("main", "robustness")}, indent=2))
        return

    if a.command == "simulate":
        print(json.dumps(A.synthetic_calibration(seed=a.seed, reps=a.reps), indent=2))
        return

    if a.command == "schedule":
        os.makedirs(a.out, exist_ok=True)
        runs, pilot = S.build_schedule(a.design), S.build_pilot(a.design)
        S.write_csv(runs, os.path.join(a.out, "schedule-{}.csv".format(a.design)))
        S.write_csv(pilot, os.path.join(a.out, "pilot-{}.csv".format(a.design)))
        print(json.dumps({"design": a.design, "seed": C.SCHEDULE_SEED,
                          "main_episodes": S.count_episodes(runs, "main"),
                          "extension_episodes": S.count_episodes(runs, "extension"),
                          "total_episodes": S.count_episodes(runs), "independent_runs": len(runs),
                          "pilot_episodes": S.count_episodes(pilot, "pilot"),
                          "robustness_pilot_episodes": S.count_episodes(pilot, "robustness_pilot"),
                          "placebo_pilot_episodes": S.count_episodes(pilot, "placebo_pilot"),
                          "paraphrase_pilot_episodes": S.count_episodes(pilot, "paraphrase_pilot")}, indent=2))
        return

    if a.command == "feasibility":
        rep = F.run_check(a.out, default_providers(), a.models.split(","), a.budget_usd)
        print(F.render_md(rep))
        return

    if a.command in ("pilot", "run", "dry-run"):
        if a.command == "dry-run":
            mock = MockProvider(mock_policy())
            providers = {"anthropic": mock, "openai": mock, "openrouter": mock}
            batch = a.batch or "dry-run"
            runs = S.build_pilot(a.design) + S.build_schedule(a.design)[: a.limit or 40]
        else:
            providers = default_providers()
            batch = a.batch or ("pilot-v1" if a.command == "pilot" else None)
            if not batch:
                ap.error("--batch is required for run")
            runs = S.build_pilot(a.design) if a.command == "pilot" else S.build_schedule(a.design)
            if a.limit:
                runs = runs[: a.limit]
        result = Runner(a.out, batch, providers, a.budget_usd, planned_runs=runs).execute(runs)
        print(json.dumps(result, indent=2))
        if a.command != "dry-run":
            return
        a.batch = batch

    if a.command in ("analyze", "dry-run"):
        batch_dir = os.path.join(a.out, a.batch)
        with open(os.path.join(batch_dir, "episodes.jsonl"), encoding="utf-8") as src:
            eps = [json.loads(l) for l in src if l.strip()]
        if any(e["component"] == "feasibility" for e in eps):
            raise SystemExit("Refusing: feasibility traces are administrative and excluded from analysis")
        is_pilot = a.pilot or a.command == "dry-run" or a.batch.startswith("pilot")
        rows = A.episode_table(eps)
        A.write_csv(rows, os.path.join(batch_dir, "episode_table.csv"))
        A.write_csv(A.secondary_by_cell(rows), os.path.join(batch_dir, "cells.csv"))
        pilot_components = ("pilot", "robustness_pilot", "placebo_pilot", "paraphrase_pilot")
        out = {"pilot_gate": A.pilot_gate([r for r in rows if r["component"] in pilot_components])}
        if not is_pilot or a.command == "dry-run":
            planned = None
            manifest_path = os.path.join(batch_dir, "manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, encoding="utf-8") as src:
                    manifest = json.load(src)
                planned_ids = set(manifest.get("planned_schedule", {}).get("run_ids", []))
                if planned_ids:
                    planned = [r for r in S.build_schedule(a.design) if r.run_id in planned_ids]
            out["report"] = A.full_report([e for e in eps if e["component"] in ("main", "extension")], planned)
        with open(os.path.join(batch_dir, "report.json"), "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(json.dumps(out, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
