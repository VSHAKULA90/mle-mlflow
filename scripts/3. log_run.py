import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import boto3
import mlflow.sklearn
from pathlib import Path
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from dotenv import load_dotenv
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, log_loss
)

load_dotenv(dotenv_path=".env")

EXPERIMENT_NAME     = "test_connection_experiment_vadim_shakula"
RUN_NAME            = os.getenv("RUN_NAME")
REGISTRY_MODEL_NAME = os.getenv("REGISTRY_MODEL_NAME")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", "http://127.0.0.1:5000"))

# ---------- 1) Данные ----------
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]

# Чтобы не ловить предупреждение про ints без NaN —
# для сигнатуры дадим сэмпл с приведением int -> float.
# На предсказания это НЕ влияет.
sig_sample = X_test.head(50).copy()
for c in sig_sample.select_dtypes(include=["int", "int64", "int32"]).columns:
    sig_sample[c] = sig_sample[c].astype("float64")

# ---------- 2) Модель ----------
model_path = "models_1sprint/model.pkl"
assert os.path.exists(model_path), f"Не найден файл модели: {model_path}"
model = joblib.load(model_path)

# ---------- 3) Предсказания ----------
prediction = model.predict(X_test)

if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X_test)[:, 1]
else:
    # аккуратный fallback: пытаемся получить score, иначе 0.5
    try:
        proba = getattr(model, "decision_function")(X_test)
        # нормируем к [0,1], если нужно
        proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-9)
    except Exception:
        proba = np.full(len(X_test), 0.5, dtype="float64")

# ---------- 4) Метрики ----------
tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
tn_n, fp_n, fn_n, tp_n = confusion_matrix(y_test, prediction, normalize="all").ravel()

metrics = {
    "auc": float(roc_auc_score(y_test, proba)),
    "precision": float(precision_score(y_test, prediction)),
    "recall": float(recall_score(y_test, prediction)),
    "f1": float(f1_score(y_test, prediction)),
    "err1": int(fp),
    "err2": int(fn),
    "err1_rate": float(fp_n),
    "err2_rate": float(fn_n),
    "logloss": float(log_loss(y_test, proba)),
}
print(metrics)

# ---------- 5) Логирование в MLflow ----------
# Гарантируем, что эксперимент есть и не удалён
client = MlflowClient()
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    if exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
    experiment_id = exp.experiment_id

pip_requirements = "requirements.txt"
# сигнатуру строим по безопасному сэмплу без int-NaN ловушки
signature = mlflow.models.infer_signature(sig_sample, prediction[: len(sig_sample)])
input_example = sig_sample.iloc[:10]
metadata = {"model_type": "monthly"}

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id):
    # 5.1 Метрики — по одной (избежать /runs/log-batch -> 503)
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # 5.2 Confusion matrix артефактом
    mlflow.log_dict({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                    artifact_file="confusion_matrix.json")

    # 5.3 Предсказания артефактом
    out = pd.DataFrame({"y_true": y_test.values, "y_pred": prediction, "proba": proba})
    out_path = "predictions.csv"
    out.to_csv(out_path, index=False)
    mlflow.log_artifact(out_path, artifact_path="predictions")

    # 5.4 Модель
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=REGISTRY_MODEL_NAME,
        signature=signature,
        input_example=input_example,
        await_registration_for=60,
        pip_requirements=pip_requirements,
        metadata=metadata,
    )
