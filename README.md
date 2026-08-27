# Tech Challenge Fase 3

API pra classificar laudo medico em `normal`, `atencao` ou `urgente`. Modelo leve (TF-IDF + regressao logistica) servido com FastAPI.

## Como rodar

Python 3.12.

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

Dataset e modelo ja estao no repo. Se quiser gerar de novo:

```bash
python model/generate_dataset.py
python model/train.py
```

Sobe a API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health`
- Predicao: `POST http://localhost:8000/predict` com `{"text": "laudo..."}`

## Docker

```bash
docker build -t triagem-laudos .
docker run --rm -p 8000:8000 triagem-laudos
```

## Testes

```bash
pytest
```

## Latencia

Com a API no ar:

```bash
python model/benchmark.py --n 200
```

## Arquitetura

O hospital precisa da classe na hora, entao a inferencia e tempo real (API HTTP). Batch nao serve pra triagem. O retreino e que pode ser job (Airflow, etapa 2).

Pra nuvem eu iria de Cloud Run no GCP: e o mesmo container Docker desta API, sobe com HTTP, escala e nao deixa VM ligada o dia inteiro. SageMaker/Vertex e overkill (e caro) pra TF-IDF. Se fosse AWS, ECR + ECS Fargate. Dado da entrega e sintetico, sem prontuario real.

Localmente a entrega e o container. Baseline no Docker (n=200): media 2.79 ms, P50 2.77 ms, P95 3.07 ms. Fora do container ficou parecido (P50 2.58 ms / P95 2.92 ms). A etapa 4 compara isso com o modelo otimizado.
