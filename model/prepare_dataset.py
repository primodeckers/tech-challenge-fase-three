"""Baixa o Medical Abstracts TC Corpus e grava data/laudos.csv."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "laudos.csv"

BASE = "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main"
TRAIN_URL = f"{BASE}/medical_tc_train.csv"
TEST_URL = f"{BASE}/medical_tc_test.csv"

LABELS = {
    1: "neoplasms",
    2: "digestive",
    3: "nervous",
    4: "cardiovascular",
    5: "general",
}


def _ler(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=60) as resp:
        raw = resp.read()
    return pd.read_csv(io.BytesIO(raw))


def preparar() -> pd.DataFrame:
    df = pd.concat([_ler(TRAIN_URL), _ler(TEST_URL)], ignore_index=True)
    df = df.rename(columns={"medical_abstract": "texto"})
    df["label"] = df["condition_label"].map(LABELS)
    df = df.dropna(subset=["texto", "label"])
    df["texto"] = df["texto"].astype(str).str.strip()
    df = df[df["texto"].str.len() > 0]
    return df[["texto", "label"]].reset_index(drop=True)


def main() -> None:
    df = preparar()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"arquivo: {OUT_PATH}")
    print(f"n={len(df)}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
