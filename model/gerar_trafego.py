"""Gera trafego continuo no /predict para popular os paineis do grafana."""

from __future__ import annotations

import argparse
import random
import time

import httpx

TEXTOS = [
    "Myocardial infarction with ST elevation and elevated cardiac enzymes.",
    "Neuropeptide Y and neuron-specific enolase levels in pheochromocytoma.",
    "Chronic gastritis with Helicobacter pylori infection confirmed by biopsy.",
    "Seizure disorder with generalized tonic-clonic episodes and EEG abnormality.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--intervalo", type=float, default=0.2)
    args = parser.parse_args()

    with httpx.Client(timeout=10.0) as client:
        for i in range(args.n):
            texto = "" if random.random() < 0.05 else random.choice(TEXTOS)
            resp = client.post(f"{args.url}/predict", json={"text": texto})
            print(i, resp.status_code)
            time.sleep(args.intervalo)


if __name__ == "__main__":
    main()
