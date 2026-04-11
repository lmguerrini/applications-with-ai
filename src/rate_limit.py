from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    updated_timestamps: list[float]
    retry_after_seconds: int


def apply_rate_limit(
    timestamps: list[float],
    *,
    now: float,
    max_requests: int,
    window_seconds: int,
) -> RateLimitResult:
    active_timestamps = [
        timestamp
        for timestamp in timestamps
        if now - timestamp < window_seconds
    ]

    if len(active_timestamps) >= max_requests:
        retry_after = max(1, int(window_seconds - (now - active_timestamps[0])) + 1)
        return RateLimitResult(
            allowed=False,
            updated_timestamps=active_timestamps,
            retry_after_seconds=retry_after,
        )

    return RateLimitResult(
        allowed=True,
        updated_timestamps=active_timestamps + [now],
        retry_after_seconds=0,
    )
