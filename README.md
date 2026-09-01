# Tech Challenge Fase 3

API que classifica abstract/laudo medico. Modelo leve (TF-IDF + regressao logistica) no FastAPI.

Dataset: **Medical Abstracts TC Corpus** (o do guia da FIAP; Kaggle / [GitHub](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)). 14.438 textos, coluna de texto + target. Classes: `neoplasms`, `digestive`, `nervous`, `cardiovascular`, `general`.

Pra baixar de novo:

```bash
python model/prepare_dataset.py
python model/train.py
```

## Como rodar

Python 3.12.

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

Sobe a API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health`
- Predicao: `POST http://localhost:8000/predict` com `{"text": "abstract..."}`

## Docker

```bash
docker build -t triagem-laudos .
docker run --rm -p 8000:8000 triagem-laudos
```

## Testes

```bash
pytest
```

Push no GitHub roda **lint** (ruff) e **pytest**.

## Airflow

Retreino: `load_data` -> `train` -> `save_model`.

```bash
docker compose up --build -d
```

UI em http://localhost:8080 (admin / admin). Dispara a DAG:

```bash
docker compose exec airflow airflow dags trigger treino_laudos
```

```bash
docker compose down
```

## Monitoramento

`docker-compose.yml` sobe a api, o prometheus e o grafana junto com o airflow:

```bash
docker compose up --build -d
```

- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin / admin), dashboard "Triagem de laudos" ja provisionado

Pra popular os graficos:

```bash
python model/gerar_trafego.py --n 200
```

3 paineis: total de requisicoes, latencia P95 e taxa de erro (4xx/5xx).

## Latencia

Com a API no ar:

```bash
python model/benchmark.py --n 200
```

Baseline no Docker (n=200): media 3.12 ms, P50 3.10 ms, P95 3.44 ms. A etapa 4 compara com o modelo otimizado.

## Arquitetura

Triagem de laudo precisa de resposta na hora, entao a inferencia e tempo real (API HTTP). Batch nao serve. O retreino fica no Airflow (DAG `treino_laudos`).

Pra nuvem eu iria de Cloud Run no GCP: e o mesmo container Docker desta API, sobe com HTTP, escala e nao deixa VM ligada o dia inteiro. SageMaker/Vertex e overkill (e caro) pra TF-IDF. Se fosse AWS, ECR + ECS Fargate. Dado e publico (abstracts), sem prontuario identificavel.
