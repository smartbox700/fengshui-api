from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"ok": True, "service": "fengshui-api"}


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/evaluate")
async def evaluate(data: dict):
    return {
        "ok": True,
        "score": 80,
        "element": "wood",
        "message": "FastAPI 서버가 정상적으로 동작 중입니다."
    }









