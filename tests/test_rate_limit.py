from src.rate_limit import apply_rate_limit


def test_apply_rate_limit_allows_request_within_limit() -> None:
    result = apply_rate_limit(
        [0.0, 10.0],
        now=20.0,
        max_requests=3,
        window_seconds=60,
    )

    assert result.allowed is True
    assert result.updated_timestamps == [0.0, 10.0, 20.0]
    assert result.retry_after_seconds == 0


def test_apply_rate_limit_blocks_when_limit_is_exceeded() -> None:
    result = apply_rate_limit(
        [0.0, 10.0, 20.0],
        now=30.0,
        max_requests=3,
        window_seconds=60,
    )

    assert result.allowed is False
    assert result.updated_timestamps == [0.0, 10.0, 20.0]
    assert result.retry_after_seconds > 0


def test_apply_rate_limit_drops_expired_timestamps() -> None:
    result = apply_rate_limit(
        [0.0, 10.0, 70.0],
        now=80.0,
        max_requests=3,
        window_seconds=60,
    )

    assert result.allowed is True
    assert result.updated_timestamps == [70.0, 80.0]


def test_apply_rate_limit_retry_after_uses_oldest_active_timestamp() -> None:
    result = apply_rate_limit(
        [41.0, 45.0],
        now=50.0,
        max_requests=2,
        window_seconds=15,
    )

    assert result.allowed is False
    assert result.retry_after_seconds == 7
