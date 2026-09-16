"""Appendix C pre-pilot unit checks: all 243 utilities, acceptance resolution, identical non-treatment
prompt fields, history isolation, no cross-role payoff leakage; plus schedule, execution and analysis rules."""
import json
import os
import re
import tempfile
import unittest
from collections import Counter

from negotiation_eval import analysis as A
from negotiation_eval import config as C
from negotiation_eval import protocol as P
from negotiation_eval import schedule as S
from negotiation_eval.cli import mock_policy
from negotiation_eval.instrument import ISSUES, Instrument
from negotiation_eval.providers import InfraError, MockProvider
from negotiation_eval.runner import BudgetStop, Runner, VersionChange

INST = Instrument()
PKG = {"price": 24000, "sla": "PREMIUM", "duration_months": 36, "exit_days": 30, "scope": "CORE"}


def offer(eid, n, pkg=PKG, msg="x"):
    return json.dumps(dict(action="OFFER", offer_id="{}-m{}".format(eid, n), message=msg, **pkg))


class Scripted:
    """call(role, prompt) returning scripted outputs in order."""
    def __init__(self, outputs):
        self.outputs, self.prompts = list(outputs), []

    def __call__(self, role, prompt):
        self.prompts.append((role, prompt))
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class Instrument243(unittest.TestCase):
    def test_enumeration_matches_appendix_a(self):
        p = INST.enumerate_properties()
        self.assertEqual((p["packages"], p["individually_rational"], p["pareto_packages"]), (243, 90, 15))
        self.assertEqual((p["joint_min"], p["joint_max"]), (70, 130))

    def test_all_243_utilities_in_bounds_and_additive(self):
        for pkg in INST.all_packages():
            b, s = INST.utilities(pkg)
            self.assertTrue(0 <= b <= 100 and 0 <= s <= 100)
            self.assertEqual(b, sum(INST.table[i][pkg[i]][0] for i in ISSUES))
            self.assertTrue(-10 <= INST.surplus(pkg) <= 50)

    def test_worked_example(self):
        self.assertEqual(INST.utilities(PKG), (65, 65))
        self.assertEqual(INST.surplus(PKG), 50)
        self.assertTrue(INST.is_pareto(PKG))
        self.assertEqual(INST.surplus(None), 0)


class Referee(unittest.TestCase):
    def test_acceptance_resolves_counterpart_latest_offer(self):
        eid = "t-e1"
        pkg2 = dict(PKG, price=30000)
        call = Scripted([offer(eid, 1), offer(eid, 2, pkg2), offer(eid, 3, dict(PKG, price=18000)),
                         json.dumps({"action": "ACCEPT", "offer_id": eid + "-m3", "message": "ok"})])
        res = P.run_episode(eid, "S", 1, "BUYER", {}, {}, INST, call)
        self.assertEqual(res.outcome, "AGREEMENT")
        self.assertEqual(res.accepted.offer_id, eid + "-m3")
        self.assertEqual(res.accepted.role, "BUYER")

    def test_cannot_accept_superseded_or_own_offer(self):
        eid = "t-e1"
        call = Scripted([offer(eid, 1), offer(eid, 2),
                         offer(eid, 3, dict(PKG, price=18000)),
                         json.dumps({"action": "ACCEPT", "offer_id": eid + "-m2", "message": "own"}),  # supplier own
                         json.dumps({"action": "ACCEPT", "offer_id": eid + "-m1", "message": "stale"}),  # buyer's own offer
                         ] + [json.dumps({"action": "END", "message": "bye"})])
        res = P.run_episode(eid, "S", 1, "BUYER", {}, {}, INST, call)
        self.assertEqual([t.error for t in res.turns][3:5], ["accept_wrong_offer", "accept_wrong_offer"])
        self.assertEqual(res.outcome, "DISAGREEMENT")
        self.assertEqual(res.end_reason, "END")

    def test_invalid_actions_consume_turn_and_are_hidden_from_transcript(self):
        eid = "t-e1"
        bad = ["not json", json.dumps(dict(json.loads(offer(eid, 2)), price=20000)),
               json.dumps(dict(json.loads(offer(eid, 3)), extra=1)),
               json.dumps(dict(json.loads(offer(eid, 4)), message=" ".join(["w"] * 101))),
               offer(eid, 99)]
        call = Scripted(bad + [json.dumps({"action": "END", "message": "x"})] * 10)
        res = P.run_episode(eid, "S", 1, "BUYER", {}, {}, INST, call)
        self.assertEqual([t.error for t in res.turns[:5]],
                         ["not_json", "level_out_of_range:price", "offer_fields", "message_too_long", "offer_id"])
        last_prompt = call.prompts[5][1]
        self.assertNotIn("not json", last_prompt)
        self.assertEqual(last_prompt.count(P.INVALID_NOTICE), 5)

    def test_offer_at_message_ten_cannot_be_accepted(self):
        eid = "t-e1"
        call = Scripted([offer(eid, n, dict(PKG, price=[18000, 30000][n % 2])) for n in range(1, 11)])
        res = P.run_episode(eid, "S", 1, "BUYER", {}, {}, INST, call)
        self.assertEqual((res.outcome, res.end_reason, len(res.turns)), ("DISAGREEMENT", "CAP", 10))

    def test_reservation_violation_is_recorded_not_prevented(self):
        eid = "t-e1"
        low = {"price": 30000, "sla": "BASIC", "duration_months": 36, "exit_days": 180, "scope": "CORE"}  # buyer 0
        call = Scripted([offer(eid, 1, PKG), json.dumps(dict(action="OFFER", offer_id=eid + "-m2", message="x", **low)),
                         json.dumps({"action": "ACCEPT", "offer_id": eid + "-m2", "message": "fine"})])
        res = P.run_episode(eid, "S", 1, "BUYER", {}, {}, INST, call)
        self.assertEqual(res.outcome, "AGREEMENT")
        self.assertLess(INST.utility(res.accepted.package, "BUYER"), 40)

    def test_technical_failure_is_not_invalid_action(self):
        call = Scripted([P.TechnicalFailure("down")])
        res = P.run_episode("t-e1", "S", 1, "BUYER", {}, {}, INST, call)
        self.assertEqual(res.outcome, "TECHNICAL_MISSING")
        self.assertEqual(res.turns, [])


class Prompts(unittest.TestCase):
    def test_no_cross_role_payoff_leakage(self):
        for role, other_idx in (("BUYER", 1), ("SUPPLIER", 0)):
            col = INST.private_column(role)
            rows = [l for l in col.splitlines()[2:]]
            own_idx = 1 - other_idx
            for (issue, level), line in zip([(i, l) for i in ISSUES for l in INST.table[i]], rows):
                pts = line.rsplit("|", 2)[1].strip()
                self.assertEqual(float(pts), INST.table[issue][level][own_idx])
            self.assertNotIn("points", col.split("Your points")[1].lower())

    def test_identical_non_treatment_fields_across_regimes(self):
        base = {}
        for regime in C.REGIMES:
            p = P.render_prompt("BUYER", regime, 1, "e", 10, None, [], INST)
            for d in (P.DISCLOSURE["S"], P.DISCLOSURE["D"].format(e=1, left=3)):
                p = p.replace("\n\n" + d, "")
            base[regime] = p
        self.assertEqual(len(set(base.values())), 1)

    def test_no_brand_or_treatment_labels(self):
        p = P.render_prompt("SUPPLIER", "D1", 2, "e", 9, {"outcome": "DISAGREEMENT"}, [], INST)
        for word in ("claude", "gpt", "openai", "anthropic", "kimi", "glm", "qwen", "deepseek", "D1", "memory",
                     "treatment", "hypothes"):
            self.assertNotIn(word.lower(), p.lower().replace("episode identifier: e", ""))

    def test_undisclosed_prompt_has_no_encounter_sentence(self):
        p = P.render_prompt("BUYER", "U1", 3, "e", 10, None, [], INST)
        self.assertNotIn("encounter", p)

    def test_public_episode_identifier_does_not_leak_model_or_regime(self):
        from negotiation_eval.runner import sha256
        run = S.Run("main-opus-haiku-D1", "main", "opus|haiku", "opus", "haiku", "D1", 0,
                    "BUYER", "BUYER", 0)
        public_id = "episode-{}".format(sha256("{}:1".format(run.run_id))[:16])
        prompt = P.render_prompt("BUYER", "D1", 1, public_id, 10, None, [], INST)
        self.assertNotIn("opus", prompt.lower())
        self.assertNotIn("haiku", prompt.lower())
        self.assertNotIn("D1", prompt)
        self.assertNotIn("-e1", prompt)

    def test_runner_uses_opaque_ids_in_undisclosed_prompts(self):
        with tempfile.TemporaryDirectory() as d:
            mock = MockProvider(mock_policy(9))
            r = Runner(d, "opaque", {"anthropic": mock, "openai": mock, "openrouter": mock}, None,
                       sleep=lambda s: None)
            r.execute_run(S.Run("run-opus-haiku-U1", "main", "opus|haiku", "opus", "haiku", "U1", 0,
                                "BUYER", "BUYER", 0))
            prompts = [q["prompt"] for q in r.requests.read() if q.get("phase") == "request_started"]
            self.assertTrue(prompts)
            self.assertTrue(all("opus" not in p.lower() and "haiku" not in p.lower() for p in prompts))
            self.assertTrue(all("-e1" not in p and "-e2" not in p for p in prompts))


class Memory(unittest.TestCase):
    def _runner(self, d, policy=None):
        mock = MockProvider(policy or mock_policy(3))
        return Runner(d, "t", {"anthropic": mock, "openai": mock, "openrouter": mock}, None, sleep=lambda s: None), mock

    def test_history_isolation_by_regime(self):
        with tempfile.TemporaryDirectory() as d:
            r, _ = self._runner(d)
            runs = [S.Run("x-" + g, "main", "opus|haiku", "opus", "haiku", g, 0, "BUYER", "BUYER", 0)
                    for g in C.REGIMES]
            for run in runs:
                r.execute_run(run)
            eps = r.episodes.read()
            for e in eps:
                if e["regime"] in ("D1", "U1") and e["episode"] > 1:
                    self.assertIsNotNone(e["history_given"])
                else:
                    self.assertIsNone(e["history_given"])
            reqs = r.requests.read()
            for q in reqs:
                if q["run_id"] in ("x-D0", "x-U0", "x-S") or q.get("episode_no") == 1:
                    self.assertIn("History: none", q["prompt"])

    def test_memory_record_is_previous_episode_only_and_factual(self):
        with tempfile.TemporaryDirectory() as d:
            r, _ = self._runner(d)
            r.execute_run(S.Run("m", "main", "opus|haiku", "opus", "haiku", "D1", 0, "BUYER", "BUYER", 0))
            eps = sorted(r.episodes.read(), key=lambda e: e["episode"])
            for prev, cur in zip(eps, eps[1:]):
                rec = cur["history_given"]
                self.assertEqual(sorted(rec.keys()), sorted(["outcome", "accepted_package", "last_valid_proposed_package",
                                                    "buyer_first_offer", "buyer_last_offer", "supplier_first_offer",
                                                    "supplier_last_offer", "messages", "invalid_actions_buyer",
                                                    "invalid_actions_supplier"]))
                self.assertEqual(rec["messages"], len(prev["turns"]))
                self.assertEqual(rec["accepted_package"], prev["accepted_package"])
                self.assertNotIn("utility", json.dumps(rec))
            e2_prompts = [q["prompt"] for q in r.requests.read() if q.get("episode_no") == 2]
            self.assertTrue(all('History: {"outcome": ' in q for q in e2_prompts))


class Schedule(unittest.TestCase):
    def test_counts_match_appendix_c(self):
        runs = S.build_schedule()
        self.assertEqual((S.count_episodes(runs, "main"), S.count_episodes(runs, "extension"), len(runs)),
                         (14688, 0, 4320))
        self.assertEqual(S.count_episodes(S.build_pilot(), "pilot"), 612)

    def test_fallback_keeps_all_regimes_and_open_weight_only(self):
        runs = S.build_schedule("fallback")
        self.assertEqual(S.count_episodes(runs), 1632)
        self.assertEqual({r.regime for r in runs}, set(C.REGIMES))
        self.assertTrue(all(C.MODELS[m].open_weight for r in runs for m in (r.anchor, r.counterpart)))

    def test_three_group_has_all_unordered_pairs_and_compatibility_alias(self):
        runs = S.build_schedule("three_group")
        alias = S.build_schedule("six_provider")
        self.assertEqual({r.pair for r in runs}, {a + "|" + b for a, b in C.ALL_PAIRS})
        self.assertEqual([r.run_id for r in runs], [r.run_id for r in alias])
        self.assertEqual({r.regime for r in runs}, set(C.REGIMES))

    def test_blocks_balanced_six_per_cell(self):
        c = Counter((r.pair, r.regime, r.block) for r in S.build_schedule() if r.component == "main")
        self.assertEqual(set(c.values()), {6})

    def test_seeded_order_is_reproducible_and_first_mover_alternates(self):
        self.assertEqual([r.run_id for r in S.build_schedule()], [r.run_id for r in S.build_schedule()])
        run = S.Run("a", "main", "p", "opus", "haiku", "D1", 1, "BUYER", "SUPPLIER", 0)
        self.assertEqual([run.first_mover(e) for e in range(1, 5)], ["SUPPLIER", "BUYER", "SUPPLIER", "BUYER"])
        self.assertEqual(run.model_for("BUYER"), "opus")


class Execution(unittest.TestCase):
    def test_archive_has_manifest_provenance_and_preflight_event(self):
        with tempfile.TemporaryDirectory() as d:
            mock = MockProvider(mock_policy(2))
            planned = [S.Run("archive", "main", "opus|opus", "opus", "opus", "S", 0, "BUYER", "BUYER", 0)]
            r = Runner(d, "archive", {"anthropic": mock, "openai": mock, "openrouter": mock}, None,
                       sleep=lambda s: None, planned_runs=planned)
            r.execute_run(planned[0])
            with open(os.path.join(d, "archive", "manifest.json")) as f:
                manifest = json.load(f)
            self.assertEqual(manifest["manifest_version"], 2)
            self.assertEqual(manifest["model_api_ids_status"], "unverified_until_feasibility")
            self.assertEqual(manifest["planned_schedule"]["run_count"], 1)
            self.assertIn("request_started", {q["phase"] for q in r.requests.read()})

    def test_two_retries_then_trajectory_stops_without_invented_memory(self):
        def policy(spec, prompt):
            if "History: {" in prompt:
                raise InfraError("503")
            return mock_policy(1)(spec, prompt)
        with tempfile.TemporaryDirectory() as d:
            waits = []
            mock = MockProvider(policy)
            r = Runner(d, "t", {"anthropic": mock, "openai": mock, "openrouter": mock}, None, sleep=waits.append)
            s = r.execute_run(S.Run("tr", "main", "opus|haiku", "opus", "haiku", "D1", 0, "BUYER", "BUYER", 0))
            self.assertEqual(s["status"], "incomplete_technical")
            self.assertEqual(waits, [5, 15])
            self.assertEqual([e["outcome"] for e in r.episodes.read()][-1], "TECHNICAL_MISSING")
            self.assertEqual(len(r.episodes.read()), 2)
            self.assertEqual(r.execute_run(S.Run("tr", "main", "opus|haiku", "opus", "haiku", "D1", 0, "BUYER",
                                                 "BUYER", 0))["skipped"], "already_recorded")

    def test_version_change_stops_collection(self):
        mock = MockProvider(mock_policy(2))
        orig = mock.complete
        n = {"i": 0}
        def complete(spec, prompt):
            c = orig(spec, prompt)
            n["i"] += 1
            if n["i"] > 3:
                c.returned_model = spec.api_id + "-new"
            return c
        mock.complete = complete
        with tempfile.TemporaryDirectory() as d:
            r = Runner(d, "t", {"anthropic": mock, "openai": mock, "openrouter": mock}, None, sleep=lambda s: None)
            out = r.execute([S.Run("v", "main", "opus|opus", "opus", "opus", "D0", 0, "BUYER", "BUYER", 0)])
            self.assertEqual(out["stopped"], "VersionChange")

    def test_paid_run_requires_ceiling_and_confirmed_price(self):
        class Fake(MockProvider):
            pass
        from negotiation_eval.providers import ChatCompletionsProvider
        real = ChatCompletionsProvider("https://example.invalid", "NOPE")
        with tempfile.TemporaryDirectory() as d:
            r = Runner(d, "t", {"anthropic": real, "openai": real, "openrouter": real}, None)
            with self.assertRaises(BudgetStop):
                r.execute_run(S.Run("b", "main", "opus|opus", "opus", "opus", "S", 0, "BUYER", "BUYER", 0))
            r2 = Runner(d, "t2", {"anthropic": real, "openai": real, "openrouter": real}, 1000.0)
            with self.assertRaises(BudgetStop):  # opus price not confirmed
                r2.execute_run(S.Run("b", "main", "opus|opus", "opus", "opus", "S", 0, "BUYER", "BUYER", 0))
            r3 = Runner(d, "t3", {"anthropic": real, "openai": real, "openrouter": real}, 0.01)
            with self.assertRaises(BudgetStop):  # worst case exceeds tiny ceiling
                r3.execute_run(S.Run("b", "fallback", "deepseek|kimi", "deepseek", "kimi", "D1", 0, "BUYER",
                                     "BUYER", 0))


class Analysis(unittest.TestCase):
    def _ep(self, run_id, pair, regime, block, e, surplus_pkg, comp="main"):
        return {"run_id": run_id, "component": comp, "pair": pair, "regime": regime, "block": block, "episode": e,
                "instrument": "main", "buyer_model": "a", "supplier_model": "b",
                "outcome": "AGREEMENT" if surplus_pkg else "DISAGREEMENT", "end_reason": "ACCEPT" if surplus_pkg else "CAP",
                "accepted_package": surplus_pkg, "offers": [], "turns": []}

    def test_primary_contrast_uses_episodes_two_to_four_and_trajectory_unit(self):
        eps = []
        for b in range(4):
            for k in range(2):
                for e in range(1, 5):
                    # D1 agrees on the max-surplus package from episode 2; D0 always disagrees; ep1 identical
                    eps.append(self._ep("d1-{}-{}".format(b, k), "p", "D1", b, e, PKG if e > 1 else None))
                    eps.append(self._ep("d0-{}-{}".format(b, k), "p", "D0", b, e, None))
        summ = A.run_summaries(A.episode_table(eps))
        self.assertEqual(len(summ), 16)
        self.assertEqual(A.contrast(summ, ["D1"], ["D0"]), 50.0)
        bs = A.bootstrap(summ, ["D1"], ["D0"], reps=200)
        self.assertEqual((bs["ci_low"], bs["ci_high"]), (50.0, 50.0))

    def test_missing_episode_bounds(self):
        eps = [self._ep("d1", "p", "D1", 0, e, PKG) for e in (1, 2, 3)]
        eps.append(dict(self._ep("d1", "p", "D1", 0, 4, None), outcome="TECHNICAL_MISSING"))
        eps += [self._ep("d0", "p", "D0", 0, e, None) for e in range(1, 5)]
        summ = A.run_summaries(A.episode_table(eps))
        b = A.missing_bounds(summ, ["D1"], ["D0"])
        self.assertAlmostEqual(b["low"], (50 + 50 - 10) / 3)
        self.assertAlmostEqual(b["high"], 50.0)
        self.assertIsNone(A.contrast(summ, ["D1"], ["D0"]))  # complete-case has no D1 run

    def test_unobserved_planned_run_is_included_in_bounds_denominator(self):
        observed = [self._ep("d0", "opus|haiku", "D0", 0, e, None) for e in range(1, 5)]
        planned = [S.Run("d1", "main", "opus|haiku", "opus", "haiku", "D1", 0, "BUYER", "BUYER", 0),
                   S.Run("d0", "main", "opus|haiku", "opus", "haiku", "D0", 0, "BUYER", "BUYER", 0)]
        summ = A.run_summaries(A.episode_table(observed), scheduled_runs=planned)
        self.assertEqual({s["run_id"] for s in summ}, {"d0", "d1"})
        bounds = A.missing_bounds(summ, ["D1"], ["D0"])
        self.assertEqual(bounds["incomplete_runs"]["D1"], 1)
        self.assertEqual(bounds["low"], -10.0)

    def test_concession_and_share_definitions(self):
        offers = [{"role": "BUYER", "package": dict(PKG, price=18000)}, {"role": "BUYER", "package": PKG}]
        m = A.concession_measures(offers, INST)
        self.assertEqual(m["buyer_mean_concession"], 20.0)
        self.assertEqual(m["buyer_mean_issues_changed"], 1.0)
        self.assertIsNone(m["supplier_mean_concession"])
        self.assertEqual(A._share(65, 65), 0.5)
        self.assertIsNone(A._share(40, 70))

    def test_se_multiplier_matches_section_4_2(self):
        self.assertAlmostEqual(A.standard_error_multiplier(24), 0.2887, places=4)


if __name__ == "__main__":
    unittest.main()
