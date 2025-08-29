#регистрация эксперемента
import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, log_loss
)

# ==== 1) Имена эксперимента/рана/модели ====
EXPERIMENT_NAME =  "test_connection_experiment_vadim_shakula"   # ваш код здесь (уникальное имя эксперимента)
RUN_NAME = "baseline_logreg_v1"
REGISTRY_MODEL_NAME = "churn_model_vadimshakula" # можно поменять на своё имя

# ==== 2) Доступ к S3 ====
assert all([
    os.environ["MLFLOW_S3_ENDPOINT_URL"],
    os.environ["AWS_ACCESS_KEY_ID"],
    os.environ["AWS_SECRET_ACCESS_KEY"]
])

# ==== 3) Подключение к Tracking/Registry ====
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_registry_uri("http://127.0.0.1:5000")

# ==== 4) Загрузка модели и тестовых данных ====
model_path = "models_1sprint/model.pkl"
model = joblib.load(model_path)

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]

# ==== 5) Предсказания ====
prediction = model.predict(X_test)                 # бинарные метки 0/1
proba = model.predict_proba(X_test)[:, 1]         # вероятности положительного класса

# ==== 6) Метрики ====
tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
metrics = {
    "auc": float(roc_auc_score(y_test, proba)),
    "precision": float(precision_score(y_test, prediction)),
    "recall": float(recall_score(y_test, prediction)),
    "f1": float(f1_score(y_test, prediction)),
    "err1": int(fp),            # FP — ошибка I рода
    "err2": int(fn),            # FN — ошибка II рода
    "logloss": float(log_loss(y_test, proba)),
}

# ==== 7) Артефакты: requirements, signature, input_example, metadata ====
pip_requirements = [
    "mlflow==2.7.1",
    "scikit-learn==1.3.1",
    "pandas",
    "numpy",
    "category-encoders",
    "catboost"
]  # ваш код здесь

# ==== 7.1 Сигнатура (int -> float64 при наличии пропусков) ====
X_sig = X_test.copy()
int_cols = X_sig.select_dtypes(include=["int32", "int64"]).columns
if len(int_cols) > 0:
    X_sig[int_cols] = X_sig[int_cols].astype("float64")
signature = infer_signature(X_sig, prediction)
input_example = X_sig.head(2)

metadata = {                                               # ваш код здесь
    "owner": "Vadim Shakula",
    "source": "sprint2-mlflow",
    **metrics
}

# ==== 8) Эксперимент: создать, если его нет ====
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = exp.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    mlflow.log_metrics(metrics)

    out = pd.DataFrame({"y_true": y_test.values, "y_pred": prediction, "proba": proba})
    out.to_csv("predictions.csv", index=False)
    mlflow.log_artifact("predictions.csv", artifact_path="predictions")

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=REGISTRY_MODEL_NAME,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        metadata=metadata,
    )

print("OK — модель зарегистрирована:", REGISTRY_MODEL_NAME)