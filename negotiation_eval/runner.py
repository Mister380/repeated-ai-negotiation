"""Execution: fresh contexts, retries, append-only JSONL archive, budget gate, version freeze.

Rules enforced (Appendix B/C, §3.6):
- every request archived with prompt hash before and response after;
- at most two identical-request retries (waits 5s, 15s); then the trajectory stops as technical missingness;
- no trajectory is resumed with an invented memory; missing runs are never replaced;
- a changed returned model version stops collection (new batch label required);
- provider-reported input above 8,000 tokens is a protocol failure, never truncated;
  a conservative character estimate is checked before dispatch;
- the budget gate reserves worst-case cost of a whole run before starting it.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
from typing import Callable, Dict, List, Optional

from . import config as C
from . import protocol as P
from .instrument import Instrument
from .providers import InfraError, MockProvider, Provider
from .schedule import Run


class BudgetStop(Exception):
    pass


class ProtocolFailure(Exception):
    pass


class VersionChange(Exception):
    pass


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class JsonlLog:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def append(self, obj: Dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read(self) -> List[Dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]


class Runner:
    def __init__(self, out_dir: str, batch: str, providers: Dict[str, Provider],
                 budget_usd: Optional[float], sleep: Callable[[float], None] = time.sleep,
                 models: Optional[Dict[str, C.ModelSpec]] = None,
                 planned_runs: Optional[List[Run]] = None):
        self.out_dir = os.path.join(out_dir, batch)
        self.batch = batch
        self.providers = providers
        self.budget = budget_usd
        self.sleep = sleep
        self.models = models or C.MODELS
        self.planned_runs = planned_runs or []
        self.requests = JsonlLog(os.path.join(self.out_dir, "requests.jsonl"))
        self.episodes = JsonlLog(os.path.join(self.out_dir, "episodes.jsonl"))
        self.runs_log = JsonlLog(os.path.join(self.out_dir, "runs.jsonl"))
        self.spent = sum(r.get("cost_usd") or 0 for r in self.requests.read())
        self.seen_versions: Dict[str, str] = {}
        for r in self.requests.read():
            if r.get("returned_model") and r["model_label"] not in self.seen_versions:
                self.seen_versions[r["model_label"]] = r["returned_model"]
        self._write_manifest()

    # ---- archive ---------------------------------------------------------------
    def _manifest(self) -> Dict:
        module_dir = os.path.dirname(__file__)
        code_hashes = {}
        for f in sorted(os.listdir(module_dir)):
            if f.endswith(".py"):
                with open(os.path.join(module_dir, f), encoding="utf-8") as src:
                    code_hashes[f] = sha256(src.read())
        return {
            "manifest_version": 2,
            "protocol_version": C.PROTOCOL_VERSION, "batch": self.batch, "created": now(),
            "python": sys.version, "platform": platform.platform(),
            "models": {k: dataclasses.asdict(v) for k, v in self.models.items()},
            "model_api_ids_status": "unverified_until_feasibility",
            "model_group_note": "open_weight is a design stratum, not a licence or weights attestation",
            "planned_schedule": {"run_count": len(self.planned_runs),
                                 "episode_count": sum(r.episodes for r in self.planned_runs),
                                 "run_ids": [r.run_id for r in self.planned_runs]},
            "settings": {k: getattr(C, k) for k in dir(C) if k.isupper()},
            "template_hashes": {
                "shared": sha256(P.SHARED_INSTRUCTION), "paraphrase": sha256(P.PARAPHRASED_INSTRUCTION),
                "case": sha256(P.CASE), "schema": sha256(P.SCHEMA),
                "disclosure": sha256(json.dumps(P.DISCLOSURE, sort_keys=True)),
                "variants": sha256(json.dumps(P.INSTRUCTION_VARIANTS, sort_keys=True)),
            },
            "code_hashes": code_hashes,
        }

    def _write_manifest(self):
        path = os.path.join(self.out_dir, "manifest.json")
        if os.path.exists(path):
            # A resumed batch must use the exact protocol/configuration that
            # created it.  Silently continuing with a changed roster or prompt
            # would make the JSONL archive internally ambiguous.
            with open(path, encoding="utf-8") as src:
                previous = json.load(src)
            current = json.loads(json.dumps(self._manifest(), default=str))
            for key in ("manifest_version", "protocol_version", "models", "settings", "template_hashes",
                        "code_hashes", "planned_schedule", "model_api_ids_status"):
                if previous.get(key) != current.get(key):
                    raise RuntimeError("manifest mismatch for existing batch: {}".format(key))
            return
        manifest = self._manifest()
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    def completed_run_ids(self) -> set:
        return {r["run_id"] for r in self.runs_log.read()}

    # ---- cost ------------------------------------------------------------------
    def _price(self, spec: C.ModelSpec, provider: Provider):
        if isinstance(provider, MockProvider):
            return 0.0, 0.0
        if spec.price_in_per_m is None or spec.price_out_per_m is None:
            raise BudgetStop("No confirmed price for {}; set it in config before paid runs".format(spec.label))
        return spec.price_in_per_m, spec.price_out_per_m

    def worst_case_run_cost(self, run: Run) -> float:
        total = 0.0
        for role in ("BUYER", "SUPPLIER"):
            spec = self.models[run.model_for(role)]
            pin, pout = self._price(spec, self.providers[spec.provider])
            per_call = (C.MAX_INPUT_TOKENS * pin + C.MAX_OUTPUT_TOKENS * pout) / 1e6
            calls = run.episodes * (C.MAX_MESSAGES // 2)
            total += per_call * calls * (1 + len(C.RETRY_WAITS_S))
        return total

    # ---- one call with retries ---------------------------------------------------
    def _call(self, run: Run, episode_id: str, counter: Dict[str, int], episode_no: Optional[int] = None):
        def call(role: str, prompt: str) -> str:
            counter["n"] += 1
            spec = self.models[run.model_for(role)]
            provider = self.providers[spec.provider]
            # Preflight is a heuristic because tokenization varies across providers.
            # Returned usage enforces the cap after dispatch; this is not a hard
            # pre-dispatch token guarantee without a verified model tokenizer.
            estimated_tokens = max(1, (len(prompt) + 2) // 3)
            if estimated_tokens > C.MAX_INPUT_TOKENS:
                raise ProtocolFailure("input estimate exceeds cap in {}".format(episode_id))
            waits = (0,) + C.RETRY_WAITS_S
            last_err = None
            for attempt, wait in enumerate(waits):
                if wait:
                    self.sleep(wait)
                rec = {"ts": now(), "batch": self.batch, "run_id": run.run_id, "episode_id": episode_id,
                       "episode_no": episode_no,
                       "message_no": counter["n"], "role": role, "model_label": spec.label, "api_id": spec.api_id,
                       "attempt": attempt, "prompt_sha256": sha256(prompt), "prompt": prompt,
                       "estimated_input_tokens": estimated_tokens,
                       "request_settings": {k: v for k, v in provider.request_body(spec, "").items()
                                            if k not in ("messages", "prompt")}}
                # Append a preflight record before dispatch.  The response is a
                # second append-only record, so a crashed process still leaves
                # evidence that a request was attempted.
                self.requests.append(dict(rec, phase="request_started"))
                try:
                    comp = provider.complete(spec, prompt)
                except InfraError as e:
                    last_err = str(e)
                    rec.update({"phase": "response", "error": last_err, "cost_usd": None})
                    self.requests.append(rec)
                    continue
                pin, pout = self._price(spec, provider)
                cost = ((comp.input_tokens or 0) * pin + (comp.output_tokens or 0) * pout) / 1e6
                self.spent += cost
                rec.update({"phase": "response", "response_text": comp.text, "returned_model": comp.returned_model,
                            "input_tokens": comp.input_tokens, "output_tokens": comp.output_tokens,
                            "request_id": comp.request_id, "cost_usd": cost, "error": None})
                self.requests.append(rec)
                if comp.input_tokens and comp.input_tokens > C.MAX_INPUT_TOKENS:
                    raise ProtocolFailure("input tokens {} > cap".format(comp.input_tokens))
                first = self.seen_versions.setdefault(spec.label, comp.returned_model or spec.api_id)
                if comp.returned_model and comp.returned_model != first:
                    raise VersionChange("{}: {} -> {}; stop and open a new batch".format(
                        spec.label, first, comp.returned_model))
                return comp.text
            raise P.TechnicalFailure(last_err)
        return call

    # ---- one run (one-shot or trajectory) -------------------------------------------
    def execute_run(self, run: Run) -> Dict:
        if run.run_id in self.completed_run_ids():
            return {"run_id": run.run_id, "skipped": "already_recorded"}
        if any(r.get("run_id") == run.run_id for r in self.requests.read() + self.episodes.read()):
            raise ProtocolFailure("partial archived trajectory {}; do not silently replay it".format(run.run_id))
        if self.budget is not None:
            if self.spent + self.worst_case_run_cost(run) > self.budget:
                raise BudgetStop("Reserving {:.2f} USD would exceed ceiling {:.2f} (spent {:.2f})".format(
                    self.worst_case_run_cost(run), self.budget, self.spent))
        elif not all(isinstance(self.providers[self.models[run.model_for(r)].provider], MockProvider)
                     for r in ("BUYER", "SUPPLIER")):
            raise BudgetStop("A monetary ceiling (--budget-usd) is required before any paid call")
        inst = Instrument(run.instrument)
        prev: Optional[P.EpisodeResult] = None
        status = "complete"
        is_placebo = run.component == "placebo_pilot"
        for e in range(1, run.episodes + 1):
            # run_id is intentionally private archive metadata and contains the
            # model labels.  The identifier shown to a model must be opaque so
            # pair, group, regime and component cannot leak through offer IDs.
            episode_id = "episode-{}".format(sha256("{}:{}".format(run.run_id, e))[:16])
            history_is_placebo = is_placebo and prev is not None
            if history_is_placebo:
                # Placebo-history pilot (edit-plan P3 item 10): episodes 2-4 get a fixed, unrelated donor
                # record instead of this trajectory's own previous episode. Pilot-only; excluded from
                # confirmatory analysis (analysis.full_report filters to component main/extension).
                record = P.PLACEBO_DONOR_RECORD
            elif run.regime in C.MEMORY_REGIMES and prev is not None:
                record = P.build_memory_record(prev)
            else:
                record = None
            histories = {"BUYER": record, "SUPPLIER": record}
            counter = {"n": 0}
            res = P.run_episode(episode_id, run.regime, e, run.first_mover(e),
                                {}, histories, inst, self._call(run, episode_id, counter, e), run.prompt_variant)
            bu, su = P.episode_utilities(res, inst)
            self.episodes.append({
                "batch": self.batch, "protocol_version": C.PROTOCOL_VERSION, **dataclasses.asdict(run),
                "episode": e, "episode_id": episode_id, "first_mover": run.first_mover(e),
                "buyer_model": run.model_for("BUYER"), "supplier_model": run.model_for("SUPPLIER"),
                "outcome": res.outcome, "end_reason": res.end_reason,
                "accepted_package": res.accepted.package if res.accepted else None,
                "offers": [dataclasses.asdict(o) for o in res.offers],
                "turns": [dataclasses.asdict(t) for t in res.turns],
                "history_given": record, "history_is_placebo": history_is_placebo,
                "history_exposure": "factual_record" if record is not None else "none",
                "buyer_utility": bu if res.outcome != "TECHNICAL_MISSING" else None,
                "supplier_utility": su if res.outcome != "TECHNICAL_MISSING" else None,
            })
            if res.outcome == "TECHNICAL_MISSING":
                status = "incomplete_technical"
                break
            prev = res
        summary = {"run_id": run.run_id, "status": status, "ts": now(), "spent_usd": self.spent}
        self.runs_log.append(summary)
        return summary

    def execute(self, runs: List[Run], stop_on: tuple = (BudgetStop, ProtocolFailure, VersionChange)) -> Dict:
        done = 0
        for run in runs:
            try:
                self.execute_run(run)
                done += 1
            except stop_on as e:
                return {"stopped": type(e).__name__, "reason": str(e), "runs_done": done, "spent_usd": self.spent}
        return {"stopped": None, "runs_done": done, "spent_usd": self.spent}
