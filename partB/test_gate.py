import pytest

from eta_mlflow import decide

PROD = {"mae": 100.0, "mae_rush": 100.0, "mae_rain": 100.0, "p95_ms": 4.0}
GOOD = {"mae": 90.0, "mae_rush": 95.0, "mae_rain": 95.0, "p95_ms": 5.0}


@pytest.mark.parametrize(
    ("metric", "criterion", "value", "passes"),
    [
        ("mae", "overall_mae", 96.99, True),
        ("mae", "overall_mae", 97.01, False),
        ("p95_ms", "latency", 9.99, True),
        ("p95_ms", "latency", 10.01, False),
        ("mae_rush", "rush_mae", 104.99, True),
        ("mae_rush", "rush_mae", 105.01, False),
        ("mae_rain", "rain_mae", 104.99, True),
        ("mae_rain", "rain_mae", 105.01, False),
    ],
)
def test_gate_boundaries(metric, criterion, value, passes):
    result = decide(PROD, GOOD | {metric: value})
    assert result["criteria"][criterion] is passes
    assert result["promote"] is passes
    assert all(
        passed for name, passed in result["criteria"].items() if name != criterion
    )
