from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Dict

app = FastAPI(
    title="fengshui-api",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# --------- Pydantic Schemas ---------
class FeatureData(BaseModel):
    direction_south: float = Field(..., ge=0, le=1, description="0.0~1.0 (남향 근접도)")
    river_distance: float = Field(..., gt=0, description="km, >0")
    road_distance: float = Field(..., gt=0, description="km, >0")
    angle_diff_to_aspect: float = Field(..., ge=0, le=180, description="0~180(deg)")

class EvaluateInput(BaseModel):
    lat: float
    lon: float
    features: FeatureData

class EvaluateOutput(BaseModel):
    ok: Literal[True]
    score: int
    element: Literal["water", "wood", "fire", "earth", "metal"]
    message: str
    details: Dict[str, float]

# --------- Scoring Logic ---------
def score_direction_south(v: float) -> float:
    return 30.0 * (v ** 0.75)

def score_river_distance(km: float) -> float:
    if km <= 0:
        return 0.0
    if km < 0.1:
        return 8.0 * (km / 0.1)
    if km <= 0.3:
        return 8.0 + 7.0 * ((km - 0.1) / 0.2)
    if km <= 0.8:
        return 15.0 + 8.0 * ((km - 0.3) / 0.5)
    if km <= 1.5:
        return 23.0 - 8.0 * ((km - 0.8) / 0.7)
    return 12.0

def score_road_distance(km: float) -> float:
    if km <= 0:
        return 0.0
    if km < 0.05:
        return 4.0 * (km / 0.05)
    if km <= 0.12:
        return 4.0 + 6.0 * ((km - 0.05) / 0.07)
    if km <= 0.6:
        return 10.0 + 8.0 * ((km - 0.12) / 0.48)
    if km <= 1.2:
        return 18.0 - 6.0 * ((km - 0.6) / 0.6)
    return 10.0

def score_angle_diff(deg: float) -> float:
    if deg < 0:
        return 0.0
    if deg <= 15:
        return 22.0 + 3.0 * (1 - deg / 15.0)
    if deg <= 30:
        return 22.0 - 6.0 * ((deg - 15) / 15)
    if deg <= 45:
        return 16.0 - 8.0 * ((deg - 30) / 15)
    if deg <= 90:
        return 8.0 - 8.0 * ((deg - 45) / 45)
    return 0.0

def element_from_score(score: float) -> str:
    if score < 40:
        return "water"
    if score < 55:
        return "wood"
    if score < 70:
        return "fire"
    if score < 85:
        return "earth"
    return "metal"

def message_from_inputs(score: float, f: FeatureData) -> str:
    hints = []
    if f.direction_south >= 0.7:
        hints.append("남향에 가까워 채광이 좋습니다")
    elif f.direction_south <= 0.3:
        hints.append("북향 성향으로 채광/환기 보완이 필요합니다")

    if 0.3 <= f.river_distance <= 0.8:
        hints.append("하천과의 거리가 조화롭습니다")
    elif f.river_distance < 0.1:
        hints.append("하천이 너무 가까워 습기/안전 보완이 필요합니다")

    if 0.12 <= f.road_distance <= 0.6:
        hints.append("도로와의 이격이 적절합니다")
    elif f.road_distance < 0.05:
        hints.append("도로 소음/먼지 유입 가능성이 큽니다")

    if f.angle_diff_to_aspect <= 15:
        hints.append("건물 방향이 이상적 방위와 잘 맞습니다")
    elif f.angle_diff_to_aspect >= 45:
        hints.append("방위 편차가 커서 내부 배치로 보완하세요")

    if not hints:
        hints.append("입지 특성을 고려한 실내 배치/색상으로 보완하세요")

    prefix = (
        "전반적으로 매우 우수합니다."
        if score >= 85 else
        "균형이 양호합니다."
        if score >= 70 else
        "보완 여지가 있습니다."
        if score >= 55 else
        "개선이 필요합니다."
    )
    return f"{prefix} " + " · ".join(hints)

# --------- Routes ---------
@app.get("/")
def root():
    return {"ok": True, "service": "fengshui-api"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/evaluate", response_model=EvaluateOutput)
def evaluate(payload: EvaluateInput):
    try:
        f = payload.features
        s_dir = score_direction_south(f.direction_south)
        s_riv = score_river_distance(f.river_distance)
        s_road = score_road_distance(f.road_distance)
        s_ang = score_angle_diff(f.angle_diff_to_aspect)

        raw = s_dir + s_riv + s_road + s_ang
        total = max(0, min(100, round(raw)))

        element = element_from_score(total)
        msg = message_from_inputs(total, f)

        return EvaluateOutput(
            ok=True,
            score=total,
            element=element,
            message=msg,
            details={
                "direction": round(s_dir, 2),
                "river": round(s_riv, 2),
                "road": round(s_road, 2),
                "angle": round(s_ang, 2),
            },
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))










