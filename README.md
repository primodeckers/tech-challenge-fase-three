# Tech Challenge Fase 3

API que classifica abstract/laudo medico. Modelo leve (TF-IDF + regressao logistica) no FastAPI.

Dataset: **Medical Abstracts TC Corpus** (o do guia da FIAP; Kaggle / [GitHub](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)). 14.438 textos, coluna de texto + target. Classes: `neoplasms`, `digestive`, `nervous`, `cardiovascular`, `general`.

O guia fala em triagem de urgencia (normal / atencao / urgente), mas esse corpus nao vem com rotulo de urgencia — vem com especialidade/condicao clinica. Optei por manter as classes originais em vez de inventar um mapeamento arbitrario pra urgencia (isso ia distorcer o rotulo real sem base clinica nenhuma). Na pratica o classificador resolve a primeira parte do problema de triagem: pra qual especialidade o laudo deveria ir. Um segundo classificador de severidade dentro de cada especialidade ficaria pra uma iteracao futura, fora do escopo desse projeto.

Pra baixar de novo:

```bash
python model/prepare_dataset.py
python model/train.py
python model/export_onnx.py
```

## Como usar

Precisa de Docker. Na pasta do repo:

```bash
docker compose up --build -d
```

Sobe API, Prometheus, Grafana e Airflow. Espera uns minutos na primeira vez (build).

| Servico | URL | usuario | senha |
|---|---|---|---|
| API | http://localhost:8000 | — | — |
| API docs | http://localhost:8000/docs | — | — |
| Prometheus | http://localhost:9090 | — | — |
| Grafana | http://localhost:3000 | `admin` | `admin` |
| Airflow | http://localhost:8081 | `admin` | `admin` |

Airflow esta na **8081** de proposito: a 8080 no host costuma estar ocupada. Grafana e Airflow pedem login; Prometheus e a API nao.

Health e predicao:

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"text\": \"Myocardial infarction with ST elevation and elevated cardiac enzymes.\"}"
```

Body: `{"text": "..."}`. Resposta: `label` + `proba` (`neoplasms`, `digestive`, `nervous`, `cardiovascular`, `general`).

No Grafana, abre o dashboard **Triagem de laudos** (ja vem provisionado). Pra os graficos mexerem:

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
python model/gerar_trafego.py --n 200
```

3 paineis: total de requisicoes, latencia P95 e taxa de erro. Prints: [docs/monitoramento.md](docs/monitoramento.md).

Retreino no Airflow (DAG `treino_laudos`: `load_data` -> `train` -> `save_model`):

```bash
docker compose exec airflow airflow dags trigger treino_laudos
```

Ou dispara pelo botao na UI.

Pra desligar:

```bash
docker compose down
```

### Sem Docker (so a API)

Python 3.12.

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Testes

```bash
pytest
```

Push no GitHub roda lint (ruff) e pytest.

### So a imagem da API

```bash
docker build -t triagem-laudos .
docker run --rm -p 8000:8000 triagem-laudos
```

## Latencia

A API usa ONNX Runtime por padrao. O joblib do sklearn fica no repo pra comparar. Inferencia local (n=200 textos do corpus, sem HTTP):

| runtime | media | P50 | P95 |
|---|---|---|---|
| sklearn | 0.76 ms | 0.75 ms | 0.94 ms |
| onnx | 0.15 ms | 0.14 ms | 0.20 ms |

HTTP local (n=200, mesmo texto no `/predict`):

| runtime | media | P50 | P95 |
|---|---|---|---|
| sklearn | 2.63 ms | 2.57 ms | 3.06 ms |
| onnx | 1.88 ms | 1.85 ms | 2.19 ms |

ONNX ficou ~5x mais rapido na inferencia. No HTTP a diferenca e menor porque entra FastAPI/rede. Pra repetir:

```bash
python model/compare_latency.py
python model/benchmark.py --n 200
```

Sklearn no HTTP: `MODEL_RUNTIME=sklearn uvicorn app.main:app --port 8001`.

## Arquitetura

Triagem de laudo precisa de resposta na hora, entao a inferencia e tempo real (API HTTP). Batch nao serve. O retreino fica no Airflow (DAG `treino_laudos`).

Pra nuvem eu iria de Cloud Run no GCP: e o mesmo container Docker desta API, sobe com HTTP, escala e nao deixa VM ligada o dia inteiro. SageMaker/Vertex e overkill (e caro) pra TF-IDF. Se fosse AWS, ECR + ECS Fargate. Dado e publico (abstracts), sem prontuario identificavel.

## Limitacoes

- Accuracy de 0.57 no holdout (5 classes reais). E um modelo leve (TF-IDF + regressao logistica) de proposito, nao um transformer — o foco do projeto e o ciclo de vida (CI/CD, retreino, monitoramento, latencia), nao o estado da arte em NLP.
- A classe `general` e ruidosa: junta abstracts que nao caem claramente nas outras 4 especialidades, entao concentra boa parte dos erros de classificacao.
- Split treino/teste e treino do modelo usam seed fixa (`random_state=42` em `model/train.py`), entao o resultado e reproduzivel rodando `python model/train.py` de novo.
