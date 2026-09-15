"""Appendix C sampling schedule: balanced blocks, randomized order with an archived seed."""
from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from typing import List

from . import config as C


@dataclass(frozen=True)
class Run:
    run_id: str
    component: str      # main | extension | pilot | robustness_pilot | placebo_pilot | paraphrase_pilot | feasibility
    pair: str           # "anchor|counterpart"
    anchor: str
    counterpart: str
    regime: str
    block: int          # 0..3, index into config.BLOCKS
    anchor_role: str
    initial_mover: str
    replicate: int
    instrument: str = "main"
    prompt_variant: str = "canonical"  # canonical | paraphrase_v1 (Appendix C paraphrase-robustness pilot)

    @property
    def episodes(self) -> int:
        return C.EPISODES[self.regime]

    def model_for(self, role: str) -> str:
        return self.anchor if role == self.anchor_role else self.counterpart

    def first_mover(self, episode_no: int) -> str:
        """First mover alternates across episodes within a trajectory; roles stay fixed."""
        if episode_no % 2 == 1:
            return self.initial_mover
        return "SUPPLIER" if self.initial_mover == "BUYER" else "BUYER"


def _cells(component, pairs, regimes, runs_per_cell, instrument="main"):
    assert runs_per_cell % len(C.BLOCKS) == 0
    per_block = runs_per_cell // len(C.BLOCKS)
    out = []
    for anchor, cp in pairs:
        for regime in regimes:
            for b, (anchor_role, mover) in enumerate(C.BLOCKS):
                for r in range(per_block):
                    rid = "{}-{}-{}-{}-b{}-r{:02d}".format(component, anchor, cp, regime, b, r)
                    out.append(Run(rid, component, anchor + "|" + cp, anchor, cp, regime, b, anchor_role, mover, r,
                                   instrument))
    return out


def build_schedule(design: str = "six_provider", seed: int = C.SCHEDULE_SEED) -> List[Run]:
    main_pairs, ext_pairs = C.design(design)
    runs = _cells("main", main_pairs, C.REGIMES, C.MAIN_RUNS_PER_CELL)
    runs += _cells("extension", ext_pairs, ("D1",), C.EXTENSION_RUNS_PER_CELL)
    random.Random(seed).shuffle(runs)
    return runs


def build_pilot(design: str = "six_provider", seed: int = C.SCHEDULE_SEED) -> List[Run]:
    """One run of every main pair-regime combination (68 episodes), blocks rotated; plus three pilot-only
    add-ons that never enter confirmatory analysis (edit-plan-2026-09-15.md P3 items 10 & 12):
    - robustness_pilot: one pair repeats D1 under the second payoff table;
    - placebo_pilot: one D1-shaped trajectory for one main pair whose episodes 2-4 receive a fixed,
      structurally identical memory record from an unrelated donor trajectory instead of their own
      previous episode (Akata et al. 2025 NHB memory-vs-context confound check);
    - paraphrase_pilot: one D1 trajectory for one main pair run under a single reworded shared instruction
      (Sclar et al. 2024 ICLR paraphrase-robustness check)."""
    main_pairs, _ = C.design(design)
    runs = []
    k = 0
    for anchor, cp in main_pairs:
        for regime in C.REGIMES:
            anchor_role, mover = C.BLOCKS[k % 4]
            runs.append(Run("pilot-{}-{}-{}".format(anchor, cp, regime), "pilot", anchor + "|" + cp, anchor, cp,
                            regime, k % 4, anchor_role, mover, 0))
            k += 1
    a, c = main_pairs[1] if len(main_pairs) > 1 else main_pairs[0]
    runs.append(Run("robustness-{}-{}-D1".format(a, c), "robustness_pilot", a + "|" + c, a, c, "D1", 0,
                    "BUYER", "BUYER", 0, instrument="robustness"))
    ap, cp0 = main_pairs[0]
    runs.append(Run("placebo-{}-{}-D1".format(ap, cp0), "placebo_pilot", ap + "|" + cp0, ap, cp0, "D1", 0,
                    "BUYER", "BUYER", 0))
    runs.append(Run("paraphrase-{}-{}-D1".format(ap, cp0), "paraphrase_pilot", ap + "|" + cp0, ap, cp0, "D1", 0,
                    "BUYER", "BUYER", 0, prompt_variant="paraphrase_v1"))
    random.Random(seed + 1).shuffle(runs)
    return runs


def count_episodes(runs: List[Run], component=None) -> int:
    return sum(r.episodes for r in runs if component is None or r.component == component)


def write_csv(runs: List[Run], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(runs[0]).keys()) + ["order"])
        w.writeheader()
        for i, r in enumerate(runs):
            row = asdict(r)
            row["order"] = i
            w.writerow(row)
