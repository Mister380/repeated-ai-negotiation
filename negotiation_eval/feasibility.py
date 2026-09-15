"""Technical feasibility check (supervisor D-2026-09-12-002).

NOT data collection and NOT the pilot. Traces go to a separate administrative directory and are
excluded from every analysis (the analysis refuses component == "feasibility").
Per provider/model it verifies: (1) access, (2) protocol implementable end to end,
(3) outputs stored and reloadable, (4) outputs comparable across providers, (5) dated version id.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from typing import Dict, List, Optional

from . import analysis as A
from . import config as C
from . import protocol as P
from .providers import Provider
from .runner import Runner
from .schedule import Run

DATE_RE = re.compile(r"(20\d{2}-?\d{2}-?\d{2}|-\d{4}$|-\d{4}[^\d])")

CASE_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[a-z0-9']+")

CONTAMINATION_PROMPT = (
    "The text below is the opening of a short case description used in a research study. If you "
    "recognise it, continue it verbatim, word for word, from memory. If you do not recognise it, "
    "write your best guess at how it continues. Reply with the continuation only.\n\n"
)


def has_dated_id(api_id: str, returned: Optional[str]) -> bool:
    return bool(DATE_RE.search(api_id or "")) or bool(returned and DATE_RE.search(returned))


def split_case(case: str = P.CASE, n_sentences: int = 2) -> tuple:
    """Split CASE into (first n_sentences, hidden remainder) for the contamination probe."""
    parts = [p for p in CASE_SENTENCE_RE.split(case.strip()) if p]
    return " ".join(parts[:n_sentences]), " ".join(parts[n_sentences:])


def _tokenize(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())


def token_overlap(response: str, remainder: str) -> float:
    """Fraction of the hidden remainder's distinct tokens that also appear in the response."""
    resp_tokens, rem_tokens = set(_tokenize(response)), set(_tokenize(remainder))
    return (len(resp_tokens & rem_tokens) / len(rem_tokens)) if rem_tokens else 0.0


def longest_common_word_ngram(response: str, remainder: str) -> int:
    """Length in words of the longest contiguous word sequence shared between response and remainder
    (dynamic-programming longest-common-substring over word-token sequences)."""
    a, b = _tokenize(response), _tokenize(remainder)
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def contamination_overlap(response: str, remainder: str) -> Dict:
    """Combined overlap score of a completion against the hidden remainder: token-overlap ratio and
    longest common word n-gram, in words and as a share of the remainder (Golchin & Surdeanu 2024 ICLR
    guided-completion probe; Sainz et al. 2023; edit-plan-2026-09-15.md P3 item 11; supervisor-notes
    ABC-1/H-04). Administrative signal only: never a claim that a model has, or has not, seen the case."""
    rem_len = len(_tokenize(remainder))
    lcng = longest_common_word_ngram(response, remainder)
    return {"token_overlap": token_overlap(response, remainder),
            "longest_common_ngram_words": lcng,
            "longest_common_ngram_ratio": (lcng / rem_len) if rem_len else 0.0}


def _single_call_cost_estimate(runner: Runner, spec: C.ModelSpec) -> float:
    provider = runner.providers[spec.provider]
    pin, pout = runner._price(spec, provider)
    return (C.MAX_INPUT_TOKENS * pin + C.MAX_OUTPUT_TOKENS * pout) / 1e6 * (1 + len(C.RETRY_WAITS_S))


def run_contamination_probe(runner: Runner, run: Run, spec: C.ModelSpec) -> Dict:
    """One extra call per model (edit-plan-2026-09-15.md P3 item 11): show the first two sentences of
    CASE and ask the model to continue verbatim; score the response against the hidden remainder.
    Respects the same no-paid-call-without-budget rule as everything else in this module."""
    if runner.budget is not None:
        try:
            if runner.spent + _single_call_cost_estimate(runner, spec) > runner.budget:
                return {"response": None, "error": "skipped: would exceed budget ceiling", "overlap": None}
        except Exception as e:
            return {"response": None, "error": "{}: {}".format(type(e).__name__, e), "overlap": None}
    prefix, remainder = split_case()
    prompt = CONTAMINATION_PROMPT + prefix
    counter = {"n": 0}
    call = runner._call(run, run.run_id + "-contam", counter)
    try:
        response = call("BUYER", prompt)
    except P.TechnicalFailure as e:
        return {"response": None, "error": str(e), "overlap": None}
    return {"response": response, "error": None, "overlap": contamination_overlap(response, remainder)}


def run_check(out_root: str, providers: Dict[str, Provider], models: List[str],
              budget_usd: Optional[float]) -> Dict:
    batch = "feasibility-" + dt.date.today().isoformat()
    runner = Runner(os.path.join(out_root, "ADMINISTRATIVE_feasibility_not_data"), batch, providers, budget_usd)
    results = {}
    for label in models:
        spec = C.MODELS[label]
        run = Run("feasibility-{}".format(label), "feasibility", label + "|" + label, label, label, "S", 0,
                  "BUYER", "BUYER", 0)
        entry = {"provider": spec.provider, "api_id": spec.api_id}
        try:
            summary = runner.execute_run(run)
            entry["accessed"] = True
            entry["protocol_implemented"] = summary.get("status") == "complete"
        except Exception as e:  # recorded, never silently dropped
            entry.update({"accessed": False, "protocol_implemented": False, "error": "{}: {}".format(type(e).__name__, e)})
        reqs = [r for r in runner.requests.read() if r["run_id"] == run.run_id]
        eps = [e for e in runner.episodes.read() if e["run_id"] == run.run_id]
        returned = next((r.get("returned_model") for r in reqs if r.get("returned_model")), None)
        entry["accessed"] = entry["accessed"] and any(r.get("error") is None for r in reqs)
        entry["stored_and_reloaded"] = bool(eps) and all("turns" in e for e in eps)
        entry["returned_model"] = returned
        entry["dated_version_id"] = has_dated_id(spec.api_id, returned)
        entry["invalid_actions"] = sum(1 for e in eps for t in e["turns"] if not t["valid"])
        entry["usage_tokens"] = {"input": sum(r.get("input_tokens") or 0 for r in reqs),
                                 "output": sum(r.get("output_tokens") or 0 for r in reqs)}
        try:
            entry["contamination_probe"] = run_contamination_probe(runner, run, spec)
        except Exception as e:  # administrative probe never fails the feasibility check itself
            entry["contamination_probe"] = {"response": None, "error": "{}: {}".format(type(e).__name__, e),
                                            "overlap": None}
        results[label] = entry
    # comparability: every stored episode scores into the same outcome schema
    eps = [dict(e, component="main") for e in runner.episodes.read()]  # relabel only for schema test
    try:
        rows = A.episode_table(eps)
        cols = {frozenset(r.keys()) for r in rows}
        comparable = len(cols) <= 1
    except Exception as e:
        comparable, rows = False, []
    for label in results:
        results[label]["comparable_schema"] = comparable and any(r["pair"] == label + "|" + label for r in rows)
    report = {"batch": batch, "scope": "administrative technical check; excluded from all analysis",
              "results": results, "spent_usd": runner.spent}
    with open(os.path.join(runner.out_dir, "feasibility_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(runner.out_dir, "feasibility_report.md"), "w") as f:
        f.write(render_md(report))
    return report


def render_md(report: Dict) -> str:
    lines = ["# Technical feasibility check — {}".format(report["batch"]), "",
             "Administrative check required by the supervisor (12 Sept 2026). Not data collection; excluded from analysis.",
             "", "| Model | Provider | Accessed | Protocol runs | Stored/reloaded | Comparable | Dated ID | Returned model | Note |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    yn = lambda b: "yes" if b else "NO"
    for label, r in report["results"].items():
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            label, r["provider"], yn(r["accessed"]), yn(r["protocol_implemented"]), yn(r["stored_and_reloaded"]),
            yn(r["comparable_schema"]), yn(r["dated_version_id"]), r.get("returned_model") or "-",
            r.get("error", "")[:80]))
    lines += ["", "Spent: {:.4f} USD".format(report["spent_usd"]), "",
              "## Contamination probe (administrative; guided-completion check, not a claim of absence)",
              "", "| Model | Token overlap | Longest common n-gram (words) | Note |",
              "| --- | --- | --- | --- |"]
    for label, r in report["results"].items():
        cp = r.get("contamination_probe") or {}
        ov = cp.get("overlap") or {}
        lines.append("| {} | {} | {} | {} |".format(
            label,
            "{:.2f}".format(ov["token_overlap"]) if ov.get("token_overlap") is not None else "-",
            ov.get("longest_common_ngram_words", "-"), (cp.get("error") or "")[:60]))
    lines += ["", "If any provider is unlikely to be accessible, amend the methodology now (not at Deliverable 3)."]
    return "\n".join(lines) + "\n"
