# Verification on 16 September 2026

- Automated suite: 56 tests passed.
- Full schedule: 4,320 independent runs, 14,688 episodes; excluded pilot 183 runs, 624 episodes.
- Mock integration: 223 runs, 748 episodes, 4,056 request-start records and 4,056 responses. No real model calls. Executed with `python3 -m negotiation_eval dry-run --out /tmp/d2-final-validation --limit 40`.
- Synthetic diagnostic: `python3 -m negotiation_eval simulate --seed 20260912 --reps 500`; output in synthetic-calibration.json.

The 100 synthetic null experiments rejected 6% (Monte Carlo interval 2.8–12.5%). The 100 known-effect experiments covered the true effect 90% (82.6–94.5%). This is an undercoverage signal for small-cell percentile intervals and a pre-registration calibration task. It must not be described as a passed coverage or real-model power check. The diagnostic uses one synthetic pair with independent normal trajectory means and does not establish calibration of the full pooled estimand. A single example interval also misses its target; it is retained rather than seed-selected away.

Required before main collection: verify endpoint availability, licenses, returned versions and request settings; run the real excluded pilot; expand calibration across bounded outcomes, unequal variances, dependence and missingness; freeze justified inference and the protocol. The repository is a tested research harness, not evidence that these steps have occurred.
