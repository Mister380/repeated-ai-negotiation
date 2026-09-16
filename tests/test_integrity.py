"""Regression tests for identification, missingness and restart integrity."""
import json
import tempfile
import unittest
from negotiation_eval import analysis as A, schedule as S
from negotiation_eval.runner import Runner
from negotiation_eval.providers import MockProvider
from negotiation_eval.cli import mock_policy

class Integrity(unittest.TestCase):
    def test_empty_required_cell_does_not_reweight(self):
        rows = [dict(pair='a|b', block=b, regime=g, mean_surplus=(None if b == 1 and g == 'D1' else 10))
                for b in (0, 1) for g in ('D0', 'D1')]
        self.assertIsNone(A.contrast(rows, ['D1'], ['D0']))
        self.assertIsNone(A.bootstrap(rows, ['D1'], ['D0'], reps=20)['ci_low'])

    def test_unstarted_trajectory_retained_in_bounds(self):
        runs = [r for r in S.build_schedule() if r.regime in ('D0', 'D1')]
        summaries = A.run_summaries([], scheduled_runs=runs)
        self.assertEqual(len(summaries), len(runs))
        self.assertIsNone(A.contrast(summaries, ['D1'], ['D0']))
        bounds = A.missing_bounds(summaries, ['D1'], ['D0'])
        self.assertEqual((bounds['low'], bounds['high']), (-60, 60))

    def test_interaction_does_not_use_pooled_sharp_null(self):
        rows = [dict(pair='a|b', block=0, regime=g, mean_surplus=v)
                for g, v in [('D0', 0), ('D1', 2), ('U0', 10), ('U1', 12)]]
        result = A.bootstrap(rows, ['D1', 'U0'], ['D0', 'U1'], reps=20)
        self.assertEqual(result['estimate'], 0)
        self.assertIsNone(result['p_value'])

    def test_prompts_opaque_and_manifest_restart_guard(self):
        with tempfile.TemporaryDirectory() as d:
            run = next(r for r in S.build_schedule() if r.regime == 'U0')
            mock = MockProvider(mock_policy())
            providers = dict(anthropic=mock, openai=mock, openrouter=mock)
            runner = Runner(d, 'check', providers, None, planned_runs=[run])
            runner.execute([run])
            requests = runner.requests.read()
            starts = [r for r in requests if r.get('phase') == 'request_started']
            self.assertTrue(starts)
            ids = {r['episode_id'] for r in starts}
            self.assertEqual(len(ids), 4)
            for r in starts:
                self.assertRegex(r['episode_id'], r'^episode-[a-f0-9]{16}$')
                self.assertNotIn(run.run_id, r['prompt'])
            Runner(d, 'check', providers, None, planned_runs=[run])
            with self.assertRaisesRegex(RuntimeError, 'planned_schedule'):
                Runner(d, 'check', providers, None, planned_runs=[])

    def test_partial_request_cannot_be_silently_replayed(self):
        with tempfile.TemporaryDirectory() as d:
            run = S.build_schedule()[0]
            mock = MockProvider(mock_policy())
            runner = Runner(d, 'partial', dict(anthropic=mock, openai=mock, openrouter=mock), None)
            runner.requests.append(dict(run_id=run.run_id, phase='request_started'))
            result = runner.execute([run])
            self.assertEqual(result['stopped'], 'ProtocolFailure')
            self.assertEqual(len(runner.requests.read()), 1)
