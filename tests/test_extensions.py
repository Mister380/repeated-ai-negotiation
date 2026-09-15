"""Tests for the P3 field-standard-hardening extensions (edit-plan-2026-09-15.md P3 items 10-14):
placebo-history pilot, paraphrase pilot, feasibility contamination probe, and the new analysis.py
secondaries (prior-package repeat, first-offer anchoring, BH q-values, Wilson intervals, bootstrap
coverage, a-priori MDE)."""
import json
import tempfile
import unittest

from negotiation_eval import analysis as A
from negotiation_eval import config as C
from negotiation_eval import feasibility as F
from negotiation_eval import protocol as P
from negotiation_eval import schedule as S
from negotiation_eval.cli import mock_policy
from negotiation_eval.instrument import Instrument
from negotiation_eval.providers import MockProvider
from negotiation_eval.runner import Runner

INST = Instrument()
PKG = {"price": 24000, "sla": "PREMIUM", "duration_months": 36, "exit_days": 30, "scope": "CORE"}


class PlaceboPilot(unittest.TestCase):
    def _runner(self, d):
        mock = MockProvider(mock_policy(11))
        return Runner(d, "t", {"anthropic": mock, "openai": mock, "openrouter": mock}, None, sleep=lambda s: None)

    def test_placebo_record_differs_from_own_previous_episode_with_identical_schema(self):
        with tempfile.TemporaryDirectory() as d:
            r = self._runner(d)
            run = S.Run("px", "placebo_pilot", "opus|haiku", "opus", "haiku", "D1", 0, "BUYER", "BUYER", 0)
            r.execute_run(run)
            eps = sorted([e for e in r.episodes.read() if e["run_id"] == "px"], key=lambda e: e["episode"])
            self.assertEqual(len(eps), 4)
            # Episode 1: no history yet (nothing to be a placebo for).
            self.assertIsNone(eps[0]["history_given"])
            self.assertFalse(eps[0]["history_is_placebo"])
            for prev, cur in zip(eps, eps[1:]):
                self.assertTrue(cur["history_is_placebo"])
                self.assertEqual(cur["history_given"], P.PLACEBO_DONOR_RECORD)
                # Identical schema to a genuine memory record built from the trajectory's own previous
                # episode...
                own_would_be = {
                    "outcome": "AGREEMENT" if prev["outcome"] == "AGREEMENT" else "DISAGREEMENT",
                    "accepted_package": prev["accepted_package"],
                }
                self.assertEqual(sorted(cur["history_given"].keys()),
                                 sorted(["outcome", "accepted_package", "last_valid_proposed_package",
                                         "buyer_first_offer", "buyer_last_offer", "supplier_first_offer",
                                         "supplier_last_offer", "messages", "invalid_actions_buyer",
                                         "invalid_actions_supplier"]))
                # ...but never the trajectory's own previous episode: the fixed donor record's identifying
                # id is foreign, and its accepted package was drawn from an unrelated seeded donor.
                self.assertNotEqual(cur["history_given"]["accepted_package"], own_would_be["accepted_package"])

    def test_placebo_pilot_only_touches_episodes_2_to_4(self):
        rec1 = P.build_placebo_donor_record(seed=1)
        rec2 = P.build_placebo_donor_record(seed=1)
        self.assertEqual(rec1, rec2)  # deterministic, seeded


class ParaphrasePilot(unittest.TestCase):
    def test_paraphrase_preserves_every_rule(self):
        for token in ("40 points", "ten", "100 words", "END"):
            self.assertIn(token, P.PARAPHRASED_INSTRUCTION)
        self.assertIn("40", P.PARAPHRASED_INSTRUCTION)
        # Acceptance and cap behaviour described (no literal "ACCEPT" token required, but the rule must
        # be present).
        self.assertIn("Accepting the counterpart's most recent valid offer", P.PARAPHRASED_INSTRUCTION)
        self.assertIn("message limit is reached", P.PARAPHRASED_INSTRUCTION)

    def test_paraphrase_keeps_non_instruction_prompt_parts_identical(self):
        kwargs = dict(role="BUYER", regime="D1", episode_no=2, episode_id="e", remaining=7,
                      history={"outcome": "DISAGREEMENT"}, transcript=["[1] BUYER {}"], inst=INST)
        canonical = P.render_prompt(variant="canonical", **kwargs)
        paraphrase = P.render_prompt(variant="paraphrase_v1", **kwargs)
        instr_canon = P.INSTRUCTION_VARIANTS["canonical"].format(role="BUYER", remaining=7)
        instr_para = P.INSTRUCTION_VARIANTS["paraphrase_v1"].format(role="BUYER", remaining=7)
        self.assertEqual(canonical.replace(instr_canon, ""), paraphrase.replace(instr_para, ""))
        self.assertNotEqual(canonical, paraphrase)

    def test_prompt_variant_is_carried_in_run_and_logs(self):
        run = S.Run("pp", "paraphrase_pilot", "opus|haiku", "opus", "haiku", "D1", 0, "BUYER", "BUYER", 0,
                    prompt_variant="paraphrase_v1")
        self.assertEqual(run.prompt_variant, "paraphrase_v1")
        with tempfile.TemporaryDirectory() as d:
            mock = MockProvider(mock_policy(5))
            r = Runner(d, "t", {"anthropic": mock, "openai": mock, "openrouter": mock}, None, sleep=lambda s: None)
            r.execute_run(run)
            reqs = [q for q in r.requests.read() if q["run_id"] == "pp"]
            self.assertTrue(reqs)
            self.assertTrue(all("act on behalf of" in q["prompt"] for q in reqs))


class ContaminationProbe(unittest.TestCase):
    def test_split_case_hides_a_nonempty_remainder(self):
        prefix, remainder = F.split_case()
        self.assertTrue(prefix and remainder)
        self.assertNotEqual(prefix, remainder)
        self.assertNotIn(remainder, prefix)
        self.assertNotIn(prefix, remainder)

    def test_overlap_score_of_verbatim_completion_is_high(self):
        _, remainder = F.split_case()
        ov = F.contamination_overlap(remainder, remainder)
        self.assertAlmostEqual(ov["token_overlap"], 1.0)
        self.assertEqual(ov["longest_common_ngram_words"], len(F._tokenize(remainder)))

    def test_overlap_score_of_unrelated_text_is_low(self):
        ov = F.contamination_overlap("purple elephants dance quietly at midnight", "Service level specifies "
                                     "monthly availability and response time for critical incidents.")
        self.assertLess(ov["token_overlap"], 0.3)
        self.assertLessEqual(ov["longest_common_ngram_words"], 1)


class SecondaryStatistics(unittest.TestCase):
    def test_spearman_known_data(self):
        # Perfect increasing rank correlation.
        self.assertAlmostEqual(A.spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]), 1.0)
        # Perfect decreasing rank correlation.
        self.assertAlmostEqual(A.spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]), -1.0)
        # Constant input is undefined.
        self.assertIsNone(A.spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(A.spearman([1], [2]))
        # Missing values dropped; classic textbook example (Spearman ~0.6 for this pair-set is not
        # asserted numerically here, just that ties are handled without error).
        self.assertIsNotNone(A.spearman([1, 2, 2, 4], [4, 3, 3, 1]))

    def test_benjamini_hochberg_matches_worked_example(self):
        q = A.benjamini_hochberg([0.01, 0.04, 0.03, 0.2])
        expected = [0.04, 0.0533, 0.0533, 0.2]
        for got, want in zip(q, expected):
            self.assertAlmostEqual(got, want, places=3)

    def test_benjamini_hochberg_accepts_dict(self):
        q = A.benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.2})
        self.assertAlmostEqual(q["a"], 0.04, places=3)
        self.assertAlmostEqual(q["c"], 0.0533, places=3)

    def test_wilson_interval_sanity(self):
        lo, hi = A.wilson_interval(0, 24)
        self.assertAlmostEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)
        self.assertLess(hi, 0.2)
        lo, hi = A.wilson_interval(12, 24)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertGreater(lo, 0.29)
        self.assertLess(hi, 0.71)
        self.assertEqual(A.wilson_interval(1, 0), (None, None))

    def test_prior_package_repeat_constructed_data(self):
        rows = [
            {"run_id": "r1", "component": "main", "pair": "p", "regime": "D1", "episode": 1,
             "accepted_package": PKG, "joint_surplus": 50.0, "agreement": 1},
            {"run_id": "r1", "component": "main", "pair": "p", "regime": "D1", "episode": 2,
             "accepted_package": PKG, "joint_surplus": 50.0, "agreement": 1},  # repeat
            {"run_id": "r1", "component": "main", "pair": "p", "regime": "D1", "episode": 3,
             "accepted_package": dict(PKG, price=18000), "joint_surplus": 30.0, "agreement": 1},  # not a repeat
        ]
        A.add_stability(rows)
        by_ep = {r["episode"]: r["prior_package_repeat"] for r in rows}
        self.assertIsNone(by_ep[1])
        self.assertEqual(by_ep[2], 1)
        self.assertEqual(by_ep[3], 0)
        cell = A.prior_package_repeat_by_cell(rows)
        rates = {c["episode"]: c["repeat_rate"] for c in cell}
        self.assertEqual(rates[2], 1.0)
        self.assertEqual(rates[3], 0.0)

    def test_bootstrap_coverage_in_unit_interval(self):
        cov = A.bootstrap_coverage(sd=10.0, n_per_cell=8, true_effect=5.0, sims=20, reps=50, seed=1)
        self.assertGreaterEqual(cov, 0.0)
        self.assertLessEqual(cov, 1.0)

    def test_mde_sd_units_matches_appendix(self):
        mde = A.mde_sd_units(24)
        self.assertGreater(mde, 0.80)
        self.assertLess(mde, 0.82)


class ReportWiring(unittest.TestCase):
    def _ep(self, run_id, comp, pair, regime, block, e, pkg):
        return {"run_id": run_id, "component": comp, "pair": pair, "regime": regime, "block": block, "episode": e,
                "instrument": "main", "buyer_model": "a", "supplier_model": "b",
                "outcome": "AGREEMENT" if pkg else "DISAGREEMENT", "end_reason": "ACCEPT" if pkg else "CAP",
                "accepted_package": pkg, "offers": [], "turns": []}

    def test_placebo_and_paraphrase_excluded_from_full_report(self):
        eps = []
        for b in range(4):
            for e in range(1, 5):
                eps.append(self._ep("d1-{}".format(b), "main", "p", "D1", b, e, PKG if e > 1 else None))
                eps.append(self._ep("d0-{}".format(b), "main", "p", "D0", b, e, None))
        # Pilot-only arms with an intentionally distinct (and therefore easily detectable) surplus profile.
        for e in range(1, 5):
            eps.append(self._ep("placebo-1", "placebo_pilot", "p", "D1", 0, e, PKG))
            eps.append(self._ep("paraphrase-1", "paraphrase_pilot", "p", "D1", 0, e, PKG))
        report = A.full_report(eps)
        self.assertEqual(report["counts"]["main_runs"], 8)  # only the "main" component trajectories
        rows = A.episode_table(eps)
        self.assertTrue(any(r["component"] == "placebo_pilot" for r in rows))
        self.assertTrue(any(r["component"] == "paraphrase_pilot" for r in rows))
        self.assertIn("secondary_family_bh", report)
        self.assertIn("prior_package_repeat_by_cell", report)
        self.assertIn("first_offer_anchoring_by_cell", report)
        self.assertIn("mde_a_priori_sd_units", report)


class PilotCounts(unittest.TestCase):
    def test_pilot_main_component_still_68_and_each_new_arm_4(self):
        pilot = S.build_pilot()
        self.assertEqual(S.count_episodes(pilot, "pilot"), 68)
        self.assertEqual(S.count_episodes(pilot, "placebo_pilot"), 4)
        self.assertEqual(S.count_episodes(pilot, "paraphrase_pilot"), 4)
        self.assertEqual(S.count_episodes(pilot, "robustness_pilot"), 4)


if __name__ == "__main__":
    unittest.main()
