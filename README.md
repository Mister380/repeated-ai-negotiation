# negotiation_eval — experiment harness for the ESSEC dissertation

Implements the protocol fixed in Deliverable 1 and the Deliverable 2 methodology proposal
(`../Deliverable 2/methodology-proposal-v2.md`, `proposal-appendices.md`). Python 3.9+, standard library only.

| Proposal section | Module |
| --- | --- |
| Appendix A: 5-issue payoff instrument, outside option 40, pilot-only robustness table | `instrument.py` |
| Appendix B: action schema, referee, prompts, disclosure, memory record, state machine | `protocol.py` |
| Appendix C: roster (six providers), pairs, fallback, blocks, seeds, caps, retries | `config.py`, `schedule.py` |
| §3.6 / App. C: fresh contexts, JSONL archive, retries, version freeze, budget gate | `runner.py`, `providers.py` |
| Supervisor D-2026-09-12-002: technical feasibility check (administrative, excluded) + contamination probe | `feasibility.py` |
| §4 / Appendix D: outcomes, concessions, D1−D0 primary, bootstrap, interaction, bounds, pilot gate | `analysis.py` |
| App. C P3 pilot add-ons: placebo-history + paraphrase-robustness arms | `protocol.py`, `schedule.py`, `runner.py` |
| App. D P3 secondaries: BH q-values, Wilson intervals, bootstrap coverage, a-priori MDE, anchoring, repeat rate | `analysis.py` |

## Commands (run from this folder)

```bash
python3 -m unittest discover -s tests -t .        # 43 pre-pilot unit checks (Appendix C)
python3 -m negotiation_eval check-instrument       # 243 / 90 IR / 15 Pareto / joint 70-130
python3 -m negotiation_eval schedule               # 1,632 + 192 = 1,824 episodes; 528 runs; pilot 68
python3 -m negotiation_eval schedule --design fallback
python3 -m negotiation_eval dry-run --out /tmp/dry --limit 528   # whole pipeline on a mock, no cost

# real calls (keys from env: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY)
python3 -m negotiation_eval feasibility --budget-usd 5
python3 -m negotiation_eval pilot --budget-usd 60
python3 -m negotiation_eval analyze --batch pilot-v1 --pilot
python3 -m negotiation_eval run --batch main-v1 --budget-usd <institutional ceiling>
python3 -m negotiation_eval analyze --batch main-v1
```

## Guarantees encoded (and tested)

- Only the acting side's payoff column enters a prompt; no brand, regime or hypothesis label.
- Non-treatment prompt text is identical across S/D0/D1/U0/U1; U regimes carry no encounter sentence.
- Memory (D1/U1, episodes 2–4) = code-generated factual record of the previous episode only; no utilities.
- Invalid actions consume a turn, are hidden from the counterpart, and are logged; no coaching or repair.
- Acceptance must reference the counterpart's most recent valid offer; a message-10 offer cannot be accepted.
- Reservation violations are recorded, never prevented.
- Two identical retries (5 s, 15 s) then technical missingness; the trajectory stops, no invented memory,
  no replacement runs.
- A changed returned model version stops collection; input > 8,000 tokens is a protocol failure.
- No paid call without `--budget-usd`; a whole run's worst-case cost is reserved before it starts; models
  with unconfirmed prices are refused.
- Trajectory = unit; D1−D0 on mean surplus of episodes 2–4, equal pair and block weights; stratified
  ("percentile") bootstrap (5,000) within pair × regime × block; −10/50 missing-episode bounds.
- Feasibility traces live under `ADMINISTRATIVE_feasibility_not_data/`; `analyze` refuses them.
- Feasibility also runs one contamination probe per model (Golchin & Surdeanu 2024; Sainz et al. 2023):
  the model is asked to continue the first two sentences of the case text verbatim; the response is scored
  against the hidden remainder (token overlap, longest common word n-gram). Administrative signal only,
  never a claim that a model has or has not seen the case.
- Pilot-only add-ons (never in the confirmatory report, always in `pilot_gate`): `placebo_pilot` — one
  D1-shaped trajectory whose episodes 2–4 get a fixed, structurally identical memory record from an
  unrelated seeded donor trajectory instead of their own previous episode (Akata et al. 2025 NHB
  memory-vs-context confound check); `paraphrase_pilot` — one D1 trajectory run under a single reworded
  shared instruction (`prompt_variant="paraphrase_v1"`, same rules and numbers; Sclar et al. 2024 ICLR).
- Exploratory secondary family (pair-specific D1−D0, interaction, D0−S, D1−S) reported with two-sided
  bootstrap p-values and one shared Benjamini–Hochberg pass; cell-level agreement/IR rates carry Wilson 95%
  intervals; `bootstrap_coverage()` simulates empirical CI coverage at an assumed pilot SD; `mde_sd_units()`
  gives the a-priori minimum detectable effect (≈0.81 SD at n=24, 80% power, α=.05), independent of any
  pilot SD estimate.
- Prior-package repeat rate (lock-in vs renewed search) and first-offer-vs-final-price Spearman anchoring
  correlation are reported per cell (App. D secondaries; stdlib-only rank correlation, ties averaged).
