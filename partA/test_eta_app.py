from uuid import UUID

import pytest
from fastapi.testclient import TestClient

import eta_app

client = TestClient(eta_app.app)
PREDICT = "/v1/models/eta:predict"
ROW = {
    "distance_km": 12,
    "n_stops": 3,
    "hour": 10,
    "weekday": 2,
    "traffic_index": 0.3,
    "rain": 0,
    "weight_kg": 5,
}


def test_valid_prediction():
    response = client.post(PREDICT, json={"instances": [ROW]})
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "v1"
    assert len(body["predictions"]) == 1
    assert isinstance(body["predictions"][0], (int, float))
    assert str(UUID(body["request_id"])) == response.headers["X-Request-ID"]


def test_batch_preserves_order():
    rows = [ROW | {"distance_km": 60}, ROW, ROW | {"distance_km": 30}]
    expected = [
        client.post(PREDICT, json={"instances": [row]}).json()["predictions"][0]
        for row in rows
    ]
    assert len(set(expected)) == len(rows)
    response = client.post(PREDICT, json={"instances": rows})
    assert response.status_code == 200
    assert response.json()["predictions"] == expected


@pytest.mark.parametrize(
    ("row", "field"),
    [
        ({key: value for key, value in ROW.items() if key != "rain"}, "rain"),
        (ROW | {"hour": 24}, "hour"),
        (ROW | {"weight_kg": "heavy"}, "weight_kg"),
    ],
)
def test_invalid_fields(row, field):
    response = client.post(PREDICT, json={"instances": [row]})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_failed"
    assert any(item["field"].endswith(field) for item in response.json()["details"])


def test_infeasible_delivery():
    row = ROW | {"distance_km": 151, "n_stops": 21}
    response = client.post(PREDICT, json={"instances": [row]})
    assert response.status_code == 422
    assert any("infeasible" in item["msg"] for item in response.json()["details"])
    request_id = response.json()["request_id"]
    assert str(UUID(request_id)) == response.headers["X-Request-ID"]


@pytest.mark.parametrize(("size", "status"), [(0, 422), (1, 200), (500, 200), (501, 422)])
def test_batch_limits(size, status):
    response = client.post(PREDICT, json={"instances": [ROW] * size})
    assert response.status_code == status
    if status == 200:
        assert len(response.json()["predictions"]) == size


@pytest.mark.parametrize(("rows", "status"), [([ROW], 200), ([], 422)])
def test_request_id_echo(rows, status):
    response = client.post(
        PREDICT, json={"instances": rows}, headers={"X-Request-ID": "mithilesh-test"}
    )
    assert response.status_code == status
    assert response.headers["X-Request-ID"] == response.json()["request_id"] == "mithilesh-test"


def test_metrics():
    assert client.get("/healthz").status_code == 200
    first = client.get("/metrics")
    second = client.get("/metrics")
    assert first.status_code == second.status_code == 200
    assert first.text == second.text
    assert 'eta_requests_total{path="/healthz",status="200"}' in first.text
    assert 'eta_request_latency_seconds_bucket{le="+Inf"}' in first.text
    assert "eta_request_latency_seconds_sum" in first.text
    assert "eta_request_latency_seconds_count" in first.text


def test_inference_error_is_generic(monkeypatch):
    def fail(_rows):
        raise RuntimeError("private model failure")

    monkeypatch.setattr(eta_app.model, "predict", fail)
    response = client.post(PREDICT, json={"instances": [ROW]})
    assert response.status_code == 500
    assert "private model failure" not in response.text
    assert response.json()["error"] == "internal_error"
    assert response.headers["X-Request-ID"] == response.json()["request_id"]
