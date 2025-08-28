import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

def main():
    # Загружаем параметры из params.yaml (если там есть target_col)
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)

    target_col = params.get("target_col", "target")

    # Загружаем данные
    df = pd.read_csv("models_1sprint/initial_data.csv")

    # Разделяем признаки и таргет
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Сплитим данные (точно как в спринте 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Создаём папку для сохранения
    os.makedirs("data/processed", exist_ok=True)

    # Сохраняем тестовые данные
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("Тестовые данные сохранены в data/processed/")

if __name__ == "__main__":
    main()
