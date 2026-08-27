"""How a measured difference is judged.

Shared by the orchestrator, which records a verdict with each result, and the
reporter, which recomputes it when rendering. Two copies of these numbers
would drift, and a threshold that differs between the thing that measures and
the thing that reports is worse than either being wrong on its own.
"""

# Lower is better for these; every other metric is higher-is-better. Changes
# are normalised so positive always means better, whichever way the metric runs.
LOWER_IS_BETTER = {"ttft_ms", "wall_ms", "peak_mem_gb"}

# Counters describing behaviour rather than speed. A change here is a
# functional failure regardless of what the timings say.
FUNCTIONAL = {"token_hit_rate", "matched_tokens", "exact_hits"}

# The smallest change worth acting on. Statistical confidence alone is not
# enough: peak memory is perfectly repeatable, so its standard error is zero
# and without a floor any nonzero delta reads as significant.
FLOOR_PCT = {
    "peak_mem_gb": 2.0,
    "decode_tps": 2.0,
    "prefill_tps": 3.0,
    "ttft_ms": 5.0,
    "wall_ms": 2.0,
}
DEFAULT_FLOOR_PCT = 3.0

# When the noise bar is this much wider than the floor, the device could not
# have detected a change worth acting on, and saying "no regression" would be
# a silent failure rather than a result.
INCONCLUSIVE_RATIO = 2.0


def floor_for(metric: str) -> float:
    return FLOOR_PCT.get(metric, DEFAULT_FLOOR_PCT)


def bar_for(metric: str, stderr_pct: float) -> float:
    """The threshold a delta must clear: confidence or relevance, whichever
    is stricter."""
    return max(stderr_pct, floor_for(metric))


def verdict(metric: str, delta: dict) -> str:
    """regressed / improved / noise / inconclusive."""
    if delta.get("change_pct") is None:
        # A counter with a zero baseline has no percentage. Switching on
        # matters only when the counter describes behaviour.
        return (
            "regressed"
            if delta.get("significant") and delta.get("functional")
            else "noise"
        )
    noise = delta.get("noise_pct", 0)
    if noise > floor_for(metric) * INCONCLUSIVE_RATIO:
        return "inconclusive"
    if abs(delta["change_pct"]) <= max(noise, floor_for(metric)):
        return "noise"
    return "improved" if delta["change_pct"] > 0 else "regressed"
