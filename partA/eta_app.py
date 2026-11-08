import os
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field, model_validator

app = FastAPI()
bundle = joblib.load(os.environ.get("ETA_MODEL", "data/eta_v1.joblib"))
model = bundle["model"]


class Delivery(BaseModel):
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def check_feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery")
        return self


class PredictionRequest(BaseModel):
    instances: list[Delivery] = Field(min_length=1, max_length=500)


def get_request_id(request: Request) -> str:
    request_id = request.headers.get("X-Request-ID")
    return request_id if request_id is not None else str(uuid4())


@app.exception_handler(RequestValidationError)
async def validation_failed(request: Request, exc: RequestValidationError):
    request_id = get_request_id(request)
    details = [
        {
            "field": ".".join(str(part) for part in error["loc"]),
            "msg": error["msg"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        headers={"X-Request-ID": request_id},
        content={
            "error": "validation_failed",
            "details": details,
            "request_id": request_id,
        },
    )


@app.get("/healthz")
def health(request: Request, response: Response):
    request_id = get_request_id(request)
    response.headers["X-Request-ID"] = request_id
    return {
        "status": "ok",
        "model_version": bundle["version"],
        "request_id": request_id,
    }


@app.post("/v1/models/eta:predict")
def predict(payload: PredictionRequest, request: Request, response: Response):
    request_id = get_request_id(request)
    rows = pd.DataFrame(
        [delivery.model_dump() for delivery in payload.instances],
        columns=bundle["feature_columns"],
    )

    try:
        predictions = [round(float(value), 1) for value in model.predict(rows)]
    except Exception:
        return JSONResponse(
            status_code=500,
            headers={"X-Request-ID": request_id},
            content={"error": "internal_error", "request_id": request_id},
        )

    response.headers["X-Request-ID"] = request_id
    return {
        "predictions": predictions,
        "model_version": bundle["version"],
        "request_id": request_id,
    }


from collections import Counter
from threading import Lock
from time import perf_counter

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.08, 0.1, 0.25, 0.5, 1.0)
metrics_lock = Lock()
request_counts = Counter()
latency_counts = Counter()
latency_sum = 0.0


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    global latency_sum

    path = request.url.path
    if path == "/metrics":
        return await call_next(request)

    start = perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = perf_counter() - start
        route = request.scope.get("route")
        route_path = getattr(route, "path", "unmatched")
        with metrics_lock:
            request_counts[route_path, status] += 1
            latency_sum += elapsed
            for bound in LATENCY_BUCKETS:
                if elapsed <= bound:
                    latency_counts[bound] += 1


@app.get("/metrics")
def metrics(request: Request):
    with metrics_lock:
        total = sum(request_counts.values())
        lines = ["# TYPE eta_requests_total counter"]
        for (path, status), count in sorted(request_counts.items()):
            lines.append(f'eta_requests_total{{path="{path}",status="{status}"}} {count}')

        lines.append("# TYPE eta_request_latency_seconds histogram")
        for bound in LATENCY_BUCKETS:
            lines.append(
                f'eta_request_latency_seconds_bucket{{le="{bound:g}"}} '
                f"{latency_counts[bound]}"
            )
        lines.extend([
            f'eta_request_latency_seconds_bucket{{le="+Inf"}} {total}',
            f"eta_request_latency_seconds_sum {latency_sum}",
            f"eta_request_latency_seconds_count {total}",
        ])

    return Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4",
        headers={"X-Request-ID": get_request_id(request)},
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_with_request_id(request: Request, exc: StarletteHTTPException):
    request_id = get_request_id(request)
    headers = dict(exc.headers or {})
    headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=exc.status_code,
        headers=headers,
        content={"detail": exc.detail, "request_id": request_id},
    )
