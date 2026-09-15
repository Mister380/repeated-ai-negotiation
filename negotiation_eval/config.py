"""Appendix C: roster, pairings, regimes, sampling and execution controls.

Roster decision (student, 14 Sept 2026; supervisor D-2026-09-12-001):
six providers = Anthropic, OpenAI, Moonshot (Kimi), Z.ai (GLM), DeepSeek, Alibaba (Qwen).
Fallback (only if funding is refused) = the four open-weight models, DeepSeek anchor, all regimes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PROTOCOL_VERSION = "d2-v2-2026-09-14"

REGIMES = ("S", "D0", "D1", "U0", "U1")
EPISODES = {"S": 1, "D0": 4, "D1": 4, "U0": 4, "U1": 4}
MEMORY_REGIMES = ("D1", "U1")
DISCLOSED_REGIMES = ("S", "D0", "D1")  # S carries its own "no further encounter" sentence

MAX_MESSAGES = 10
MAX_WORDS = 100
MAX_INPUT_TOKENS = 8000
MAX_OUTPUT_TOKENS = 4096
TEMPERATURE = 0.7
REQUEST_TIMEOUT_S = 120
RETRY_WAITS_S = (5, 15)  # at most two identical-request retries

SCHEDULE_SEED = 20260910
ILLUSTRATION_SEED = 20260911
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260912

MAIN_RUNS_PER_CELL = 24
EXTENSION_RUNS_PER_CELL = 12

# Pilot pass criteria (Appendix C)
PILOT_MAX_INVALID_OVERALL = 0.05
PILOT_MAX_INVALID_PER_MODEL = 0.10
PILOT_MAX_CAP_REACHED_PER_MODEL = 0.25


@dataclass(frozen=True)
class ModelSpec:
    label: str
    provider: str            # anthropic | openai | openrouter
    api_id: str
    family: str
    tier: str
    open_weight: bool
    dated_id: bool           # advertised dated snapshot (verified at feasibility check)
    supports_temperature: bool = True
    extra: Dict[str, object] = field(default_factory=dict)  # reasoning settings, recorded verbatim
    price_in_per_m: Optional[float] = None    # USD; None = must be set before paid runs
    price_out_per_m: Optional[float] = None


MODELS: Dict[str, ModelSpec] = {m.label: m for m in [
    ModelSpec("opus", "anthropic", "claude-opus-5", "anthropic", "high", False, False,
              extra={"thinking": {"type": "adaptive"}, "effort": "medium"}),
    ModelSpec("haiku", "anthropic", "claude-haiku-4-5-20251001", "anthropic", "compact", False, True,
              extra={"thinking": "off"}),
    ModelSpec("gpt55", "openai", "gpt-5.5-2026-04-23", "openai", "high", False, True,
              supports_temperature=False, extra={"reasoning_effort": "medium"}),
    ModelSpec("luna", "openai", "gpt-5.6-luna", "openai", "compact", False, False,
              supports_temperature=False, extra={"reasoning_effort": "medium"}),
    ModelSpec("deepseek", "openrouter", "deepseek/deepseek-v4-pro-0813", "deepseek", "open", True, True,
              price_in_per_m=0.579, price_out_per_m=1.738),
    ModelSpec("qwen", "openrouter", "qwen/qwen3.8-max-0902", "qwen", "open", True, True,
              price_in_per_m=2.00, price_out_per_m=6.00),
    ModelSpec("kimi", "openrouter", "moonshotai/kimi-k3", "kimi", "open", True, False,
              price_in_per_m=2.40, price_out_per_m=12.00),
    ModelSpec("glm", "openrouter", "z-ai/glm-5.3", "glm", "open", True, False,
              price_in_per_m=1.40, price_out_per_m=4.40),
]}
# NOTE: frontier prices are placeholders to be confirmed from provider pricing pages
# before any paid run; the budget gate refuses to dispatch a model whose price is None.

Pair = Tuple[str, str]  # (anchor, counterpart)

MAIN_PAIRS: List[Pair] = [("opus", "opus"), ("opus", "haiku"), ("opus", "gpt55"), ("opus", "luna")]
EXTENSION_PAIRS: List[Pair] = [("opus", "kimi"), ("opus", "glm"), ("opus", "deepseek"), ("opus", "qwen")]
FALLBACK_PAIRS: List[Pair] = [("deepseek", "deepseek"), ("deepseek", "kimi"), ("deepseek", "glm"), ("deepseek", "qwen")]

# Balance blocks: (which role the anchor plays, who moves first in episode 1)
BLOCKS = [("BUYER", "BUYER"), ("BUYER", "SUPPLIER"), ("SUPPLIER", "BUYER"), ("SUPPLIER", "SUPPLIER")]


def design(name: str):
    """Return (main_pairs, extension_pairs) for a named design."""
    if name == "six_provider":
        return MAIN_PAIRS, EXTENSION_PAIRS
    if name == "fallback":
        return FALLBACK_PAIRS, []
    raise ValueError("design must be 'six_provider' or 'fallback'")
