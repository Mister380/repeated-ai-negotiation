# negotiation_eval

This folder contains the reproducible evaluation harness for the ESSEC repeated cross-model negotiation study. It uses a fixed five-issue payoff instrument, a referee-driven JSON action protocol, fresh request contexts, and append-only JSONL archives.

The confirmatory design has eight candidate model labels and all 36 unordered model pairings, including self-pairings. Every pair is run in all five regimes (`S`, `D0`, `D1`, `U0`, `U1`) with 24 independent trajectories per pair/regime. The four role and first-mover blocks are balanced at six trajectories each. This is 4,320 independent trajectories and 14,688 episodes. The default design label is `three_group`; `six_provider` is accepted as a compatibility alias for the same all-pairs schedule.

The three analysis groups are Anthropic, OpenAI, and `open_weight`. The last name is a design stratum only and does not attest to a model's licence or weight availability. Candidate API identifiers are recorded as unverified until the administrative feasibility check confirms the returned version. No model or provider name enters a negotiation prompt: public episode identifiers are opaque hashes, and the payoff table contains only the acting side's points.

The primary estimand is the D1−D0 difference in mean joint surplus over episodes 2–4, with the trajectory as the independent unit. It gives every unordered model pair equal weight and every block equal weight within pair. Technical missingness is kept separate from substantive disagreement; complete-case estimates are accompanied by bounds that fill every missing scheduled episode with the instrument's −10 to 50 surplus range. A percentile bootstrap supplies intervals. Exploratory D1−D0 p-values use within-pair/block label randomization under a sharp exchangeability null; the factorial interaction and D0/S and D1/S comparisons remain descriptive because their nulls require different treatment of the repeated episode aggregation. Secondary results include the six group-pair strata, with an equal-stratum sensitivity that is explicitly unestimable when any required stratum is absent.

Repeated-episode rates use run-clustered bootstrap intervals. Wilson intervals are reserved for independent one-episode cells. Pilot-only placebo-history, paraphrase, and robustness arms are archived and gated separately; they never enter the confirmatory report. Feasibility traces are administrative and are refused by the analysis command.

Useful no-call checks from this folder:

```bash
python3 -m unittest discover -s tests -t .
python3 -m negotiation_eval check-instrument
python3 -m negotiation_eval schedule
python3 -m negotiation_eval simulate --seed 20260912 --reps 500
python3 -m negotiation_eval dry-run --out /tmp/negotiation-eval-dry --limit 40
```

`simulate` is a deterministic synthetic calibration. It makes no network or paid calls and reports a known-null contrast, a known-effect confidence-interval check, an explicit missingness-bound example, and 100 independent synthetic experiments reporting null rejection and interval coverage with Monte Carlo uncertainty. A single interval can miss the true effect; these diagnostics do not establish power for real model outcomes. `dry-run` exercises the archive, referee, memory, and report wiring with the mock provider.

The main modules are:

| Module | Responsibility |
| --- | --- |
| `config.py` | roster, six group-pair strata, regimes, caps, and seeds |
| `schedule.py` | all-pairs schedule, balanced blocks, pilot additions, and invariant checks |
| `protocol.py` | action schema, prompt assembly, referee, memory record, and state machine |
| `runner.py` | provider calls, retries, version freeze, spend controls, and archived manifest |
| `analysis.py` | outcome table, trajectory summaries, primary contrast, sensitivity bounds, bootstrap, randomization, and report |
| `feasibility.py` | administrative access/version/schema check and contamination probe |
