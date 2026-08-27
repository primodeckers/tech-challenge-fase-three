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
