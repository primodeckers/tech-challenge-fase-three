"""Mede latencia do POST /predict (P50 e P95)."""

from __future__ import annotations

import argparse
import statistics
import time

import httpx

TEXTO = (
    "Myocardial infarction with ST elevation and elevated cardiac enzymes. "
    "Coronary angiography showed occlusion of the left anterior descending artery."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    times_ms: list[float] = []
    with httpx.Client(timeout=10.0) as client:
        for _ in range(10):
            client.post(args.url, json={"text": TEXTO}).raise_for_status()
        for _ in range(args.n):
            t0 = time.perf_counter()
            resp = client.post(args.url, json={"text": TEXTO})
            resp.raise_for_status()
            times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms.sort()
    p50 = statistics.median(times_ms)
    p95 = times_ms[int(0.95 * (len(times_ms) - 1))]
    media = statistics.mean(times_ms)
    print(f"n={args.n}")
    print(f"media_ms={media:.2f}")
    print(f"p50_ms={p50:.2f}")
    print(f"p95_ms={p95:.2f}")


if __name__ == "__main__":
    main()
