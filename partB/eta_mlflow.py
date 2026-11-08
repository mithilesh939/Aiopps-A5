import os
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "eta-model"
EXPERIMENT_NAME = "mithilesh-eta"


def predict_bundle(bundle, model_input):
    features = model_input.loc[:, bundle["feature_columns"]]
    predictions = bundle["model"].predict(features)
    return pd.DataFrame({
        "eta_minutes": predictions.round(1),
        "model_version": bundle["version"],
    })


class EtaModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.bundle = joblib.load(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        return predict_bundle(self.bundle, model_input)


def register():
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise RuntimeError("Set MLFLOW_TRACKING_URI to Mithilesh's SQLite URI first")

    artifact_dir = ROOT / ".local" / "mlflow-artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME, artifact_location=artifact_dir.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id

    fresh = pd.read_csv(ROOT / "data" / "eta_fresh.csv")
    requirements = [
        f"{name}=={version(name)}"
        for name in ("mlflow", "cloudpickle", "numpy", "pandas", "scikit-learn", "joblib")
    ]

    for bundle_version, alias in (("v1", "prod"), ("v2", "candidate")):
        bundle_path = ROOT / "data" / f"eta_{bundle_version}.joblib"
        bundle = joblib.load(bundle_path)
        example = fresh[bundle["feature_columns"]].head(5)
        signature = infer_signature(example, predict_bundle(bundle, example))

        with mlflow.start_run(experiment_id=experiment_id, run_name=bundle_version):
            logged = mlflow.pyfunc.log_model(
                name=f"eta-{bundle_version}",
                python_model=EtaModel(),
                artifacts={"bundle": str(bundle_path)},
                input_example=example,
                signature=signature,
                pip_requirements=requirements,
                code_paths=[str(Path(__file__).resolve())],
            )

        registered = mlflow.register_model(logged.model_uri, MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, alias, registered.version)
        print(f"{alias}: registry version {registered.version}, bundle {bundle_version}")


def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0,
           max_slice_regression=0.05):
    criteria = {
        "overall_mae": cand["mae"] <= prod["mae"] * (1 - min_rel_gain),
        "latency": cand["p95_ms"] <= max_p95_ms,
        "rush_mae": cand["mae_rush"] <= prod["mae_rush"] * (1 + max_slice_regression),
        "rain_mae": cand["mae_rain"] <= prod["mae_rain"] * (1 + max_slice_regression),
    }
    return {"criteria": criteria, "promote": all(criteria.values())}


def promotion_gate(**thresholds):
    from time import perf_counter

    import numpy as np
    from eta_data import FEATURES, make_data

    fresh = make_data(2000, seed=99)
    inputs = fresh[FEATURES]
    rush = fresh["hour"].between(7, 9) | fresh["hour"].between(16, 19)
    rain = fresh["rain"] == 1
    metrics = {}

    for alias in ("prod", "candidate"):
        model = mlflow.pyfunc.load_model(f"models:/eta-model@{alias}")
        predictions = model.predict(inputs)["eta_minutes"].to_numpy()
        errors = (fresh["eta_minutes"] - predictions).abs()

        timings_ms = []
        for index in range(200):
            row = inputs.iloc[[index]]
            start = perf_counter()
            model.predict(row)
            timings_ms.append((perf_counter() - start) * 1000)

        metrics[alias] = {
            "mae": float(errors.mean()),
            "mae_rush": float(errors[rush].mean()),
            "mae_rain": float(errors[rain].mean()),
            "p95_ms": float(np.percentile(timings_ms, 95)),
        }

    return {
        **metrics,
        **decide(metrics["prod"], metrics["candidate"], **thresholds),
    }


if __name__ == "__main__":
    register()
