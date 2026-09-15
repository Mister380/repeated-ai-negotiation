"""Appendix A: service-contract payoff instrument.

Five issues x three levels = 243 packages. Utilities are additive, private per role,
and the outside option is 40 points per side. Everything here is deterministic.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ISSUES = ("price", "sla", "duration_months", "exit_days", "scope")
OUTSIDE_OPTION = 40.0
ROLES = ("BUYER", "SUPPLIER")

# issue -> level -> (buyer points, supplier points, human-readable term)
MAIN_TABLE: Dict[str, Dict[object, Tuple[float, float, str]]] = {
    "price": {
        18000: (40, 0, "EUR 18,000 per year"),
        24000: (20, 20, "EUR 24,000 per year"),
        30000: (0, 40, "EUR 30,000 per year"),
    },
    "sla": {
        "BASIC": (0, 5, "99.0% availability and 8-hour response"),
        "STANDARD": (12.5, 2.5, "99.5% availability and 4-hour response"),
        "PREMIUM": (25, 0, "99.9% availability and 1-hour response"),
    },
    "duration_months": {
        12: (5, 0, "12 months"),
        24: (2.5, 12.5, "24 months"),
        36: (0, 25, "36 months"),
    },
    "exit_days": {
        180: (0, 10, "180 days exit notice"),
        90: (10, 5, "90 days exit notice"),
        30: (20, 0, "30 days exit notice"),
    },
    "scope": {
        "CORE": (0, 20, "Core support only"),
        "QUARTERLY": (5, 10, "Core plus quarterly reporting"),
        "MONTHLY": (10, 0, "Core plus monthly reporting and review"),
    },
}

# Pilot-only robustness table (Appendix C): same levels, reordered issue priorities.
# Buyer now weights SLA most, supplier weights duration most. Enumerated like the main table.
ROBUSTNESS_TABLE: Dict[str, Dict[object, Tuple[float, float, str]]] = {
    "price": {18000: (25, 0, MAIN_TABLE["price"][18000][2]), 24000: (12.5, 12.5, MAIN_TABLE["price"][24000][2]),
              30000: (0, 25, MAIN_TABLE["price"][30000][2])},
    "sla": {"BASIC": (0, 10, MAIN_TABLE["sla"]["BASIC"][2]), "STANDARD": (20, 5, MAIN_TABLE["sla"]["STANDARD"][2]),
            "PREMIUM": (40, 0, MAIN_TABLE["sla"]["PREMIUM"][2])},
    "duration_months": {12: (10, 0, "12 months"), 24: (5, 20, "24 months"), 36: (0, 40, "36 months")},
    "exit_days": {180: (0, 5, MAIN_TABLE["exit_days"][180][2]), 90: (7.5, 2.5, MAIN_TABLE["exit_days"][90][2]),
                  30: (15, 0, MAIN_TABLE["exit_days"][30][2])},
    "scope": {"CORE": (0, 20, "Core support only"), "QUARTERLY": (5, 10, "Core plus quarterly reporting"),
              "MONTHLY": (10, 0, "Core plus monthly reporting and review")},
}

TABLES = {"main": MAIN_TABLE, "robustness": ROBUSTNESS_TABLE}

Package = Dict[str, object]


@dataclass(frozen=True)
class Instrument:
    name: str = "main"

    @property
    def table(self):
        return TABLES[self.name]

    def levels(self, issue: str) -> List[object]:
        return list(self.table[issue].keys())

    def is_valid_package(self, pkg: Package) -> bool:
        return set(pkg.keys()) == set(ISSUES) and all(pkg[i] in self.table[i] for i in ISSUES)

    def utility(self, pkg: Package, role: str) -> float:
        idx = 0 if role == "BUYER" else 1
        return float(sum(self.table[i][pkg[i]][idx] for i in ISSUES))

    def utilities(self, pkg: Optional[Package]) -> Tuple[float, float]:
        if pkg is None:
            return OUTSIDE_OPTION, OUTSIDE_OPTION
        return self.utility(pkg, "BUYER"), self.utility(pkg, "SUPPLIER")

    def surplus(self, pkg: Optional[Package]) -> float:
        """Joint surplus above disagreement; zero on disagreement (Appendix D)."""
        if pkg is None:
            return 0.0
        b, s = self.utilities(pkg)
        return b + s - 2 * OUTSIDE_OPTION

    def all_packages(self) -> List[Package]:
        return [dict(zip(ISSUES, combo)) for combo in itertools.product(*(self.levels(i) for i in ISSUES))]

    def is_pareto(self, pkg: Package) -> bool:
        b, s = self.utilities(pkg)
        for other in self.all_packages():
            ob, os_ = self.utilities(other)
            if ob >= b and os_ >= s and (ob > b or os_ > s):
                return False
        return True

    def private_column(self, role: str) -> str:
        """Render only the acting side's payoff column (no cross-role leakage)."""
        idx = 0 if role == "BUYER" else 1
        lines = ["| Issue | Allowed value | Your points |", "| --- | --- | --- |"]
        for issue in ISSUES:
            for level, row in self.table[issue].items():
                lines.append("| {} | {} ({}) | {} |".format(issue, level, row[2], _fmt(row[idx])))
        return "\n".join(lines)

    def enumerate_properties(self) -> Dict[str, float]:
        pkgs = self.all_packages()
        joint = [sum(self.utilities(p)) for p in pkgs]
        ir = [p for p in pkgs if min(self.utilities(p)) >= OUTSIDE_OPTION]
        return {
            "packages": len(pkgs),
            "individually_rational": len(ir),
            "pareto_packages": sum(1 for p in pkgs if self.is_pareto(p)),
            "joint_min": min(joint),
            "joint_max": max(joint),
            "utility_min": min(min(self.utilities(p)) for p in pkgs),
            "utility_max": max(max(self.utilities(p)) for p in pkgs),
        }


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)
