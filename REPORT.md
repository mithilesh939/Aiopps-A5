# SwiftParcel ETA Service

**AIOps · Module 4 · Assignment 5**

**Student:** Mithilesh  
**Roll number:** da24b048

**Machine:** Intel Core i7-13620H, 10 physical cores, 16 logical CPUs, and 14 GiB RAM reported by Linux. All runs used CPU only. Ray was started with two logical CPUs.

## Part A — REST service

The FastAPI service loads the v1 Ridge bundle once per worker. It validates all seven features, accepts batches of 1–500 deliveries, and rejects deliveries over 150 km with more than 20 stops. Predictions preserve input order and are rounded to one decimal place. The app echoes a supplied request ID or generates one, returns structured validation errors, and reports request counts and a latency histogram. Metrics scrapes do not increment the counters.

The app passed 14 pytest cases covering valid predictions, batch order and limits, invalid fields, infeasible deliveries, request IDs, metrics, and generic inference errors. Two additional manual TestClient checks verified request IDs on HTTP 404 and 405. The load generator ran for four seconds at each concurrency, with batch size 1 and Uvicorn access logging disabled in both configurations. All requests in the following runs succeeded.

| Concurrency | 1 worker req/s | p50 / p95 / p99 (ms) | 2 workers req/s | p50 / p95 / p99 (ms) |
|---:|---:|---:|---:|---:|
| 1 | 652.5 | 1.5 / 1.8 / 2.3 | 641.5 | 1.5 / 1.8 / 2.1 |
| 2 | 852.5 | 2.2 / 3.0 / 3.8 | 1052.0 | 1.8 / 2.4 / 2.9 |
| 4 | 890.0 | 4.3 / 5.6 / 6.7 | 1642.8 | 2.3 / 3.3 / 4.2 |
| 8 | 829.5 | 9.3 / 12.4 / 15.3 | 1516.8 | 5.1 / 7.9 / 9.8 |
| 16 | 823.5 | 19.2 / 25.0 / 30.3 | 1510.5 | 10.3 / 14.0 / 17.7 |

Saturation appeared around concurrency 4. With a closed-loop client, Little’s Law relates average requests in flight to throughput times mean response time. With one worker, increasing concurrency from 4 to 16 multiplied the client’s concurrency by four while throughput fell from 890.0 to 823.5 req/s. In the approximation that each client has one request in flight, Little’s Law gives a mean time of about 4.5 ms at concurrency 4 and 19.4 ms at concurrency 16. These are estimates from concurrency and throughput. The measured p95 rose from 5.6 to 25.0 ms over the same interval. The highest observed zero-error throughput under the 80 ms p95 target was 890.0 req/s with one worker. Adding a second worker gave independent serving capacity and raised the observed peak to 1642.8 req/s, a 1.85× improvement.

### Batch cost at concurrency 1

| Rows per request | Requests/s | p95 per request (ms) | Throughput-derived ms per row |
|---:|---:|---:|---:|
| 1 | 652.5 | 1.8 | 1.53 |
| 100 | 483.8 | 2.4 | 0.0207 |

The per-row cost is 1000 ÷ (requests per second × rows per request). Batch 100 increased request-level p95 by 0.6 ms while processing about 74 times as many rows per second. Batching spreads fixed HTTP and model-call overhead across more rows.

p95 and p99 describe how long requests near the slow end of the distribution take. The mean can hide those slow responses when most requests are fast.

## Part B — MLflow registry, gate, and shadow

| Registry model | Registry version | Alias | Bundle |
|---|---:|---|---|
| eta-model | 1 | prod | v1 |
| eta-model | 2 | candidate | v2 |

Both models were logged to a local SQLite-backed registry with a signature, an input example, and pinned Python package requirements. The custom pyfunc returns rounded ETA values and the bundle version. Eight promotion-gate tests passed, covering values just inside and outside all four decision boundaries. The gate evaluated 2,000 fresh rows generated with seed 99 and timed 200 single-row pyfunc predictions per alias.

| Metric | Prod v1 | Candidate v2 |
|---|---:|---:|
| Overall MAE (min) | 11.9553 | 6.1049 |
| Rush-hour MAE (min) | 19.6799 | 8.5795 |
| Rain MAE (min) | 11.6227 | 6.2853 |
| Local p95, default run (ms) | 0.980 | 0.952 |
| Local p95, 60% gain run (ms) | 0.978 | 0.966 |

The default gate required at least 3% overall MAE improvement, candidate p95 at or below 10 ms, and at most 5% regression in each slice. All four criteria passed, so the decision was `promote=True`. With a 60% minimum gain, the overall MAE criterion failed because the measured improvement was 48.9%. Both slices and latency still passed, giving `promote=False`.

| Shadow metric, 200 paired rows | Result |
|---|---:|
| Mean absolute prediction difference | 9.484 min |
| Differences greater than 10 min | 70/200 (35.0%) |
| Prod HTTP p95 | 3.436 ms |
| Candidate HTTP p95 | 3.269 ms |

I would send 10% of customers to v2 in a monitored canary because it improved overall and slice MAE with similar HTTP latency. Since 35% of paired predictions differed by over 10 minutes, I would monitor the effect on customer-facing ETAs before expanding traffic. An already-running MLflow server keeps the model it resolved and loaded at startup, so moving an alias affects subsequent model loads. A rolling replacement can start ready replicas with the new model before draining the old replicas. A parallel deployment can warm a second service and then switch traffic once it passes readiness checks.

## Part C — Ray Serve sticky canary

| Verification | Result |
|---|---|
| 20% weight, 1,000 distinct customers | 200/1,000 assigned to v2 |
| Repeat requests | 200/200 kept their arm |
| Identical redeployment | 200/200 kept their arm |
| Weight increased to 50% | All 200 original v2 customers stayed on v2, with 510/1,000 on v2 overall |
| Header override | 50/50 v1-assigned customers received v2 |

The router hashes the customer ID with SHA-256 and compares the bucket to the weight cutoff. The `X-Beta: 1` header routes an individual request to v2. Python salts its built-in string hash per process, so the same ID can map to a different bucket after a restart. SHA-256 gives a stable bucket, and raising the cutoff keeps the original candidate buckets selected. Stickiness keeps each customer’s ETA experience consistent across requests and keeps treatment groups stable for customer-level analysis.

I set the target to two ongoing requests per replica so sustained queueing can trigger scaling at a modest load. A five-second scale-up delay lets brief bursts pass before adding a replica. A 30-second scale-down delay keeps capacity available through short dips. The cap of three candidate replicas bounds CPU use, and at maximum scale the router and model replicas reserve 1.5 CPUs total.

## Part D — Release smoke test

The v1 smoke test passed all six checks. The baseline ETA was 35.3 minutes, rising to 97.2 for greater distance and 61.0 for heavier traffic. An empty batch returned HTTP 422. Over 20 requests, p95 was 1.84 ms against an 80 ms limit.

| Abort metric for a 10% v2 canary | Trigger |
|---|---|
| v2 HTTP 5xx rate | Above 1% over five minutes, with at least 100 requests |
| v2 prediction latency | p95 above 80 ms over five minutes, with at least 100 requests |
| Invalid v2 ETA | Any missing, non-finite, or negative ETA in a successful response within five minutes |
| Paired accuracy after delivery | v2 MAE above 110% of v1 MAE over a rolling 24 hours, with at least 100 paired outcomes |

## Reproduction and scope

Use a fresh copy of the supplied starter, copy the submitted Python files from their part directories to its root, and use a Python environment with the starter requirements. Run `python eta_data.py` from that root to generate the bundles and CSVs. For Part A run `python -m pytest -q test_eta_app.py` and serve `eta_app:app` with Uvicorn using `data/eta_v1.joblib`. Use the starter README to generate the benchmark sample and run the load sweep. Use one and two workers with access logging disabled for the comparison.

For Part B, create `.local`, set `MLFLOW_TRACKING_URI="sqlite:///$PWD/.local/mithilesh-mlflow.db"`, and run `python eta_mlflow.py` once for a fresh registry. Run `python -m pytest -q test_gate.py` and call `promotion_gate()` with defaults and with `min_rel_gain=0.60`. Serve the two aliases on 5101 and 5102 and send the first 200 fresh feature rows to each `/invocations` endpoint as paired single-row requests. For Part C, use a short private `RAY_TMPDIR` under `/tmp` and run `python verify_ray.py`. For Part D, pass the full prediction URL and expected model version to `python smoke_eta.py`.

**Not tested:** Ray autoscaling under sustained load, a live 10% canary, and the Part D smoke check against v2. The first benchmark attempt returned errors on every request and was excluded. The benchmark tables use the successful reruns.

**Sources:** The assignment-provided `eta_data.py`, `load_test.py`, and starter README were used. OpenAI Codex assisted with implementation, verification commands, and report drafting.
