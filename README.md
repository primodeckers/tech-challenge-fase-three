# Tech Challenge Fase 3

API que classifica abstract/laudo medico. Modelo leve (TF-IDF + regressao logistica) no FastAPI.

Dataset: **Medical Abstracts TC Corpus** (o que o enunciado da FIAP sugere; Kaggle / [GitHub](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)). 14.438 textos, coluna de texto + target. Classes: `neoplasms`, `digestive`, `nervous`, `cardiovascular`, `general`.

O enunciado fala em triagem de urgencia (normal / atencao / urgente), mas esse corpus nao vem com rotulo de urgencia — vem com especialidade/condicao clinica. Optei por manter as classes originais em vez de inventar um mapeamento arbitrario pra urgencia (isso ia distorcer o rotulo real sem base clinica nenhuma). Na pratica o classificador resolve a primeira parte do problema de triagem: pra qual especialidade o laudo deveria ir. Um segundo classificador de severidade dentro de cada especialidade ficaria pra uma iteracao futura, fora do escopo desse projeto.

Video STAR (ate 5 min): https://www.youtube.com/watch?v=mnWF6gW0ro0
Arquivo no repo: [docs/Triagem_Médica_Automatizada.mp4](docs/Triagem_Médica_Automatizada.mp4)

Dados da entrega (GitHub, video, URLs, senhas): [ENTREGA.txt](ENTREGA.txt)

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

O enunciado pede pra escolher estrategia de deploy em nuvem (AWS, Azure ou GCP) olhando batch vs tempo real. A entrega em si continua local (`docker compose`); o que vai abaixo e a decisao — o mesmo container desta API, so que em servico gerenciado.

### O que o hospital precisa

Triagem de laudo e classificacao de texto na porta de entrada: chega o abstract/laudo, o sistema devolve a classe (`neoplasms`, `digestive`, `nervous`, `cardiovascular`, `general`) na mesma chamada. Se a inferencia for um job de madrugada (Spark, fila, CSV processado de madrugada), o medico espera e a triagem perde o sentido. Por isso a inferencia e **tempo real**: FastAPI, `POST /predict`, P95 aqui na casa de ~2 ms com ONNX. Nao precisa GPU nem cluster.

O retreino e o contrario. Nao tem que caber no caminho do POST. Roda quando entra lote novo de laudos, valida o CSV, treina TF-IDF + regressao logistica e grava o artefato. Isso e **batch**. No lab e a DAG `treino_laudos` no Airflow (`load_data` -> `train` -> `save_model`).

Ou seja: o sistema e hibrido. Tempo real na ponta, batch no ciclo do modelo. Escolher "so batch" ou "so real-time" ignora metade do problema.

Tem ainda o terceiro padrao da disciplina, **serverless** (Lambda / Cloud Functions / Azure Functions). Caberia se o artefato fosse um pickle minusculo e o trafego fosse esporadico. Aqui a imagem ja leva sklearn + ONNX Runtime + o `.onnx`; Lambda tem teto de tamanho e o cold start e pior que o de um container quente. Pra hospital, primeira request lenta no plantao e risco. Entao serverless puro eu descartaria pra inferencia. Container HTTP (Cloud Run / Fargate / Container Apps) e o meio-termo: sobe a mesma imagem do Docker, escala, e ainda da pra deixar 1 instancia minima.

### Como eu compararia AWS, Azure e GCP

Os tres resolvem. A pergunta nao e "qual nuvem e melhor", e "qual servico encaixa num classificador leve em CPU, HTTP, retreino periodico, sem VM o dia inteiro".

**AWS.** Inferencia: ECR guarda a imagem, ECS Fargate sobe o container sem eu gerenciar EC2. API Gateway na frente se precisar de auth e throttle. Retreino: um task Fargate ou AWS Batch na mesma imagem de treino; EventBridge dispara de madrugada. SageMaker Endpoint eu nao usaria: e plataforma de ML, billing de instancia de endpoint, pra TF-IDF nao ganha latencia nenhuma que o FastAPI ja nao tenha. EC2 24h tambem nao: paga ocioso no plantao fraco e vira patch de SO. Lambda so faria sentido com modelo bem menor do que o que esta em `artifacts/`.

**Azure.** O par equivalente e Azure Container Registry + Container Apps (HTTP, escala, CPU). Jobs do Container Apps (ou Azure Batch) pro retreino. Azure ML / AKS eu deixaria de fora: AKS e Kubernetes, o enunciado nao pede orquestrador de cluster; Azure ML e o SageMaker da casa. VM no App Service ou numa VM classica tem o mesmo vicio da EC2 ligada.

**GCP.** Artifact Registry + Cloud Run na API; Cloud Run Jobs no retreino. Cloud Scheduler dispara o job. Se o hospital ja pagasse Airflow gerenciado, seria Composer (Airflow no GCP) no lugar do Jobs, reusando a DAG que ja existe. Vertex AI Prediction e o analogo de SageMaker/Azure ML: faz sentido pra modelo grande, nao pra este. Compute Engine o dia inteiro, mesma critica de custo.

Monitoracao nativa em cada um (CloudWatch, Azure Monitor, Cloud Monitoring) cobriria latencia e erro da API. Neste trabalho o equivalente portavel e Prometheus raspando `/metrics` + Grafana com os 3 paineis — e o que o enunciado manda rodar no compose. Em nuvem eu manteria as metricas da API e mandaria tambem pro monitor nativo; nao jogaria o Prometheus fora so porque a cloud tem dashboard.

CI/CD tambem mapeia: o GitHub Actions daqui ja faz lint + pytest; o passo extra em nuvem seria `docker build` + push no registry (ECR / ACR / Artifact Registry) e deploy do mesmo tag. O enunciado nao exige esse build no workflow, mas e o encaixe natural.

### Escolha: GCP Cloud Run

Eu iria de **GCP Cloud Run** pra API. Motivos, neste cenario:

1. A unidade de deploy ja e o Dockerfile desta API. Cloud Run recebe o container, expoe HTTPS, escala com request. Nao reescreve FastAPI pra Lambda.
2. FinOps: escala a zero se ninguem chama. Hospital de referencia nao e loja 24h com mil RPS, mas tambem nao e zero; o ponto e nao pagar GCE ocioso. Em producao eu travaria `min instances = 1` e `max instances` baixo (CPU, modelo leve, P95 de poucos ms — uma instancia segura muita chamada). Assim o plantao nao leva cold start na primeira triagem.
3. CPU basta. Latencia local com ONNX ja esta bem abaixo do que um humano percebe. GPU / Vertex nao mudam a nota clinica e incham a conta.
4. Retreino: Cloud Run Jobs com o mesmo codigo de `model/train.py` (e o export ONNX). Composer so se valesse a pena pagar Airflow gerenciado; no lab o Airflow no compose ja demonstra a DAG.
5. Dado e publico (Medical Abstracts TC Corpus), sem prontuario. Mesmo assim, em nuvem: HTTPS obrigatorio, IAM na frente da API, secret fora da imagem, registry privado. No lab a API nao tem auth de proposito — e demonstracao local.

Se o time fosse mais AWS, a traducao direta e **ECR + ECS Fargate** (e nao SageMaker). Azure: **ACR + Container Apps**. A arquitetura nao muda, so o nome do servico. O que eu nao escolheria em nenhum dos tres: VM ligada 24h, cluster Kubernetes, endpoint de plataforma de ML, e batch como unico modo de inferencia.

## Limitacoes

- Accuracy de 0.57 no holdout (5 classes reais). E um modelo leve (TF-IDF + regressao logistica) de proposito, nao um transformer — o foco do projeto e o ciclo de vida (CI/CD, retreino, monitoramento, latencia), nao o estado da arte em NLP.
- A classe `general` e ruidosa: junta abstracts que nao caem claramente nas outras 4 especialidades, entao concentra boa parte dos erros de classificacao.
- Split treino/teste e treino do modelo usam seed fixa (`random_state=42` em `model/train.py`), entao o resultado e reproduzivel rodando `python model/train.py` de novo.
