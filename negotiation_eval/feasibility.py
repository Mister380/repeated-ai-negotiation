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
from .providers import Provider
from .runner import Runner
from .schedule import Run

DATE_RE = re.compile(r"(20\d{2}-?\d{2}-?\d{2}|-\d{4}$|-\d{4}[^\d])")


def has_dated_id(api_id: str, returned: Optional[str]) -> bool:
    return bool(DATE_RE.search(api_id or "")) or bool(returned and DATE_RE.search(returned))


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
    lines += ["", "Spent: {:.4f} USD".format(report["spent_usd"]),
              "", "If any provider is unlikely to be accessible, amend the methodology now (not at Deliverable 3)."]
    return "\n".join(lines) + "\n"
