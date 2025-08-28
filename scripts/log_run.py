import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    log_loss
)

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]

# ---- 1) Загрузка модели (pipeline) ----
# сначала пробуем canonical имя, затем имя из первого спринта
model_path_candidates = [
    "models_1sprint/model.pkl",
]

model_path = next((p for p in model_path_candidates if os.path.exists(p)), None)
assert model_path is not None, f"Не найден файл модели. Ожидал один из: {model_path_candidates}"

model = joblib.load(model_path)

# ---- 2) Предсказания модели ----
# бинарные предсказания
prediction = model.predict(X_test)

# вероятности положительного класса (для ROC-AUC и logloss)
# у CatBoostClassifier есть predict_proba
proba = model.predict_proba(X_test)[:, 1]

# ---- 3) Метрики ----
metrics = {}

# ROC-AUC (ТОЛЬКО по вероятностям)
metrics["auc"] = roc_auc_score(y_test, proba)

# Классические метрики по классам
metrics["precision"] = precision_score(y_test, prediction)
metrics["recall"]    = recall_score(y_test, prediction)
metrics["f1"]        = f1_score(y_test, prediction)

# Матрица ошибок: порядок элементов -> TN, FP, FN, TP
tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
# Если хочешь нормированные доли от всех наблюдений:
tn_n, fp_n, fn_n, tp_n = confusion_matrix(y_test, prediction, normalize="all").ravel()

# Ошибка I рода (FP), ошибка II рода (FN)
metrics["err1"] = int(fp)        # число FP
metrics["err2"] = int(fn)        # число FN
# (опционально) нормированные доли
metrics["err1_rate"] = float(fp_n)  # доля FP от всех объектов
metrics["err2_rate"] = float(fn_n)  # доля FN от всех объектов

# Logloss — тоже по вероятностям!
metrics["logloss"] = log_loss(y_test, proba)

print(metrics)
