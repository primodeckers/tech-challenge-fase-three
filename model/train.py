"""Treina o classificador de abstracts medicos (TF-IDF + regressao logistica)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "laudos.csv"
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
SEED = 42


def treinar() -> None:
    df = pd.read_csv(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(
        df["texto"],
        df["label"],
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"],
    )
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=15000,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=SEED),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    print(classification_report(y_test, y_pred))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    treinar()
