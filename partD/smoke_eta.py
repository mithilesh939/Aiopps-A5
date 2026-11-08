import argparse
import math
import sys
from time import perf_counter

import requests

ROW = {
    "distance_km": 12,
    "n_stops": 3,
    "hour": 10,
    "weekday": 2,
    "traffic_index": 0.3,
    "rain": 0,
    "weight_kg": 5,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run_checks(url, expected_version, max_p95_ms):
    with requests.Session() as session:
        def predict(row):
            response = session.post(url, json={"instances": [row]}, timeout=5)
            response.raise_for_status()
            body = response.json()
            require(body["model_version"] == expected_version, "unexpected model version")
            values = body["predictions"]
            require(len(values) == 1, "expected exactly one prediction")
            eta = values[0]
            require(type(eta) in (int, float) and math.isfinite(eta) and eta >= 0,
                    "expected a finite, nonnegative ETA")
            return eta

        baseline = predict(ROW)
        print(f"PASS valid prediction: {baseline:.1f} min")
        print(f"PASS model version: {expected_version}")

        farther = predict(ROW | {"distance_km": 36})
        require(farther >= baseline, "longer distance shortened the ETA")
        print(f"PASS distance sanity: {baseline:.1f} -> {farther:.1f} min")

        heavier_traffic = predict(ROW | {"traffic_index": 0.9})
        require(heavier_traffic >= baseline, "heavier traffic shortened the ETA")
        print(f"PASS traffic sanity: {baseline:.1f} -> {heavier_traffic:.1f} min")

        invalid = session.post(url, json={"instances": []}, timeout=5)
        require(invalid.status_code == 422, f"expected HTTP 422, got {invalid.status_code}")
        print("PASS invalid input: HTTP 422")

        timings_ms = []
        for _ in range(20):
            start = perf_counter()
            predict(ROW)
            timings_ms.append((perf_counter() - start) * 1000)

        rank = math.ceil(0.95 * len(timings_ms))
        p95 = sorted(timings_ms)[rank - 1]
        require(p95 <= max_p95_ms, f"p95 {p95:.2f} ms exceeded {max_p95_ms:g} ms")
        print(f"PASS latency: p95 {p95:.2f} ms over 20 requests (limit {max_p95_ms:g} ms)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Full prediction endpoint URL")
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-p95-ms", type=float, default=80)
    args = parser.parse_args()

    try:
        run_checks(args.url, args.expect_version, args.max_p95_ms)
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
