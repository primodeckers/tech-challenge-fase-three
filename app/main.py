from fastapi import FastAPI

app = FastAPI(title="Triagem de laudos")


@app.get("/health")
def health():
    return {"status": "ok"}
