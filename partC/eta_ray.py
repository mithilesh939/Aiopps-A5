from hashlib import sha256
from pathlib import Path
from time import sleep

import joblib
import pandas as pd
from fastapi import FastAPI, Header
from pydantic import BaseModel
from ray import serve

DATA = Path(__file__).resolve().parent / "data"
api = FastAPI()


class CanaryRequest(BaseModel):
    customer_id: str
    instances: list[dict]


@serve.deployment(ray_actor_options={"num_cpus": 0.25})
class EtaModel:
    def __init__(self, version, extra_work_ms):
        self.bundle = joblib.load(DATA / f"eta_{version}.joblib")
        self.extra_work_ms = extra_work_ms

    def predict(self, instances):
        if self.extra_work_ms > 0:
            sleep(self.extra_work_ms / 1000)
        rows = pd.DataFrame(instances, columns=self.bundle["feature_columns"])
        predictions = self.bundle["model"].predict(rows)
        return {
            "predictions": [round(float(value), 1) for value in predictions],
            "model_version": self.bundle["version"],
        }


@serve.deployment(ray_actor_options={"num_cpus": 0.25})
@serve.ingress(api)
class EtaRouter:
    def __init__(self, prod, candidate, weight):
        self.prod = prod
        self.candidate = candidate
        self.cutoff = int(weight * (1 << 64))

    @api.post("/v1/models/eta:predict")
    async def predict(self, payload: CanaryRequest, x_beta: str | None = Header(None)):
        digest = sha256(payload.customer_id.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big")
        use_candidate = x_beta == "1" or bucket < self.cutoff
        handle = self.candidate if use_candidate else self.prod
        result = await handle.predict.remote(payload.instances)
        return {**result, "arm": "v2" if use_candidate else "v1"}


def build(weight=0.1, extra_work_ms=0.0):
    prod = EtaModel.options(name="eta-prod", num_replicas=2).bind("v1", 0.0)
    candidate = EtaModel.options(
        name="eta-candidate",
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 3,
            "target_ongoing_requests": 2,
            "upscale_delay_s": 5,
            "downscale_delay_s": 30,
        },
    ).bind("v2", extra_work_ms)
    return EtaRouter.bind(prod, candidate, weight)
