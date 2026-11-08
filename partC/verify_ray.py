import socket

import ray
import requests
from ray import serve

from eta_ray import build

HOST = "127.0.0.1"
PORT = 8016
APP_NAME = "mithilesh-eta"
URL = f"http://{HOST}:{PORT}/v1/models/eta:predict"
ROW = {
    "distance_km": 12,
    "n_stops": 3,
    "hour": 10,
    "weekday": 2,
    "traffic_index": 0.3,
    "rain": 0,
    "weight_kg": 5,
}


def arm(session, customer_id, beta=False):
    response = session.post(
        URL,
        json={"customer_id": customer_id, "instances": [ROW]},
        headers={"X-Beta": "1"} if beta else {},
        timeout=20,
    )
    response.raise_for_status()
    result = response.json()
    assert result["arm"] in ("v1", "v2"), result
    assert result["model_version"] == result["arm"], result
    assert len(result["predictions"]) == 1, result
    return result["arm"]


def main():
    with socket.socket() as probe:
        probe.bind((HOST, PORT))

    ray.init(address="local", num_cpus=2, include_dashboard=False)
    try:
        serve.start(http_options={"host": HOST, "port": PORT})
        serve.run(build(weight=0.2), name=APP_NAME)

        with requests.Session() as session:
            ids = [f"mithilesh-customer-{index}" for index in range(1000)]
            initial = {customer: arm(session, customer) for customer in ids}
            old_candidates = {customer for customer in ids if initial[customer] == "v2"}
            share = len(old_candidates) / len(ids)
            assert 0.16 <= share <= 0.24, share
            print(f"PASS allocation: {len(old_candidates)}/1000 to v2 ({share:.1%})")

            for customer in ids[:200]:
                assert arm(session, customer) == initial[customer], customer
            print("PASS repeat requests: 200/200 kept their arm")

            serve.delete(APP_NAME)
            serve.run(build(weight=0.2), name=APP_NAME)
            for customer in ids[:200]:
                assert arm(session, customer) == initial[customer], customer
            print("PASS identical redeploy: 200/200 kept their arm")

            serve.run(build(weight=0.5), name=APP_NAME)
            expanded = {customer for customer in ids if arm(session, customer) == "v2"}
            assert old_candidates <= expanded
            assert len(expanded) > len(old_candidates)
            print(
                f"PASS weight increase: all {len(old_candidates)} original v2 customers "
                f"stayed on v2, now {len(expanded)}/1000 on v2"
            )

            prod_ids = [customer for customer in ids if customer not in expanded]
            assert len(prod_ids) >= 50
            for customer in prod_ids[:50]:
                assert arm(session, customer, beta=True) == "v2", customer
            print("PASS beta override: 50/50 v1 customers received v2")
    finally:
        try:
            serve.shutdown()
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
