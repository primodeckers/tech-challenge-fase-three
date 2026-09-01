# Tech Challenge Fase 3

API que classifica abstract/laudo medico. Modelo leve (TF-IDF + regressao logistica) no FastAPI.

Dataset: **Medical Abstracts TC Corpus** (o do guia da FIAP; Kaggle / [GitHub](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)). 14.438 textos, coluna de texto + target. Classes: `neoplasms`, `digestive`, `nervous`, `cardiovascular`, `general`.

O guia fala em triagem de urgencia (normal / atencao / urgente), mas esse corpus nao vem com rotulo de urgencia — vem com especialidade/condicao clinica. Optei por manter as classes originais em vez de inventar um mapeamento arbitrario pra urgencia (isso ia distorcer o rotulo real sem base clinica nenhuma). Na pratica o classificador resolve a primeira parte do problema de triagem: pra qual especialidade o laudo deveria ir. Um segundo classificador de severidade dentro de cada especialidade ficaria pra uma iteracao futura, fora do escopo desse projeto.

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

Prints da stack rodando: [docs/monitoramento.md](docs/monitoramento.md).

## Latencia

Com a API no ar:

```bash
python model/benchmark.py --n 200
```

Baseline no Docker (n=200): media 3.12 ms, P50 3.10 ms, P95 3.44 ms. A etapa 4 compara com o modelo otimizado.

## Arquitetura

Triagem de laudo precisa de resposta na hora, entao a inferencia e tempo real (API HTTP). Batch nao serve. O retreino fica no Airflow (DAG `treino_laudos`).

Pra nuvem eu iria de Cloud Run no GCP: e o mesmo container Docker desta API, sobe com HTTP, escala e nao deixa VM ligada o dia inteiro. SageMaker/Vertex e overkill (e caro) pra TF-IDF. Se fosse AWS, ECR + ECS Fargate. Dado e publico (abstracts), sem prontuario identificavel.

## Limitacoes

- Accuracy de 0.57 no holdout (5 classes reais). E um modelo leve (TF-IDF + regressao logistica) de proposito, nao um transformer — o foco do projeto e o ciclo de vida (CI/CD, retreino, monitoramento, latencia), nao o estado da arte em NLP.
- A classe `general` e ruidosa: junta abstracts que nao caem claramente nas outras 4 especialidades, entao concentra boa parte dos erros de classificacao.
- Split treino/teste e treino do modelo usam seed fixa (`random_state=42` em `model/train.py`), entao o resultado e reproduzivel rodando `python model/train.py` de novo.
