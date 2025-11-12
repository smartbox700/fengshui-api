from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Dict, List, Optional
from supabase import create_client, Client
import os
import math

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(
    title="fengshui-api",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# -------------------------------------------------
# CORS
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://fengshui-api.onrender.com",
    ],
    allow_origin_regex=r"^https://.*\.vercel\.app$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Supabase (옵션)
# -------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Optional[Client] = (
    create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
)


async def _insert_log(record: dict):
    """Supabase에 로그 남기기 (실패해도 API 응답엔 영향 X)."""
    try:
        if not supabase:
            return
        supabase.table("api_logs").insert(record).execute()
    except Exception:
        pass


def _build_log_record(
    path: str,
    request: Optional[Request],
    payload: Optional[dict],
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    score: Optional[int] = None,
    element: Optional[str] = None,
    details: Optional[dict] = None,
) -> dict:
    return {
        "path": path,
        "ip": (request.client.host if request and request.client else None),
        "user_agent": (request.headers.get("user-agent") if request else None),
        "lat": lat,
        "lon": lon,
        "score": score,
        "element": element,
        "details": details,
        "payload": payload,
    }

# -------------------------------------------------
# 기본 라우트
# -------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "fengshui-api"}


@app.get("/health")
def health():
    return {"ok": True}


# -------------------------------------------------
# 점수 계산용 함수들
# -------------------------------------------------
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


def message_from_inputs(score: float, f: "FeatureData") -> str:
    hints: List[str] = []
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
        "전반적으로 매우 우수합니다." if score >= 85
        else "균형이 양호합니다." if score >= 70
        else "보완 여지가 있습니다." if score >= 55
        else "개선이 필요합니다."
    )
    return f"{prefix} " + " · ".join(hints)


# -------------------------------------------------
# 모델들
# -------------------------------------------------
class FeatureData(BaseModel):
    direction_south: float = Field(..., ge=0, le=1)
    river_distance: float = Field(..., gt=0)
    road_distance: float = Field(..., gt=0)
    angle_diff_to_aspect: float = Field(..., ge=0, le=180)


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


class ColorSuggestInput(BaseModel):
    element: Literal["water", "wood", "fire", "earth", "metal"]
    style: Optional[Literal["modern", "minimal", "vintage", "lux"]] = "modern"


class ColorSuggestOutput(BaseModel):
    ok: Literal[True]
    element: Literal["water", "wood", "fire", "earth", "metal"]
    style: Literal["modern", "minimal", "vintage", "lux"]
    palette: List[str]
    suggestions: Dict[str, str]
    message: str


class ReportInput(EvaluateInput):
    style: Optional[Literal["modern", "minimal", "vintage", "lux"]] = "modern"


class ReportOutput(BaseModel):
    ok: Literal[True]
    score: int
    element: Literal["water", "wood", "fire", "earth", "metal"]
    summary: str
    recommendations: List[str]
    palette: List[str]
    suggestions: Dict[str, str]
    details: Dict[str, float]


class ScoreOutput(BaseModel):
    ok: Literal[True]
    score: int
    element: Literal["water", "wood", "fire", "earth", "metal"]


# -------------------------------------------------
# 팔레트 관련
# -------------------------------------------------
BASE_PALETTES: Dict[str, List[str]] = {
    "water": ["#0EA5E9", "#38BDF8", "#93C5FD", "#111827"],
    "wood": ["#16A34A", "#86EFAC", "#A3E635", "#374151"],
    "fire": ["#EF4444", "#F97316", "#FCA5A5", "#1F2937"],
    "earth": ["#A16207", "#D6A354", "#F5E6CC", "#4B5563"],
    "metal": ["#94A3B8", "#CBD5E1", "#E5E7EB", "#0F172A"],
}


def _clamp(v: int) -> int:
    return max(0, min(255, v))


def _adjust_hex(hex_color: str, factor: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = _clamp(int(int(hex_color[0:2], 16) * factor))
    g = _clamp(int(int(hex_color[2:4], 16) * factor))
    b = _clamp(int(int(hex_color[4:6], 16) * factor))
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def build_palette(element: str, style: str) -> List[str]:
    base = BASE_PALETTES[element]
    if style == "minimal":
        factors = [1.25, 1.05, 0.9, 0.7, 0.5, 0.35, 0.2]
    elif style == "vintage":
        factors = [0.9, 0.8, 0.7, 1.0, 1.1, 0.6, 0.45]
    elif style == "lux":
        factors = [0.85, 0.7, 0.55, 1.0, 1.15, 0.4, 0.25]
    else:
        factors = [1.15, 1.0, 0.85, 0.7, 0.55, 0.4, 0.25]
    c0, c1, c2, cN = base[0], base[1], base[2], base[3]
    seeds = [c0, c1, c2, cN, c0, c1, cN]
    return [_adjust_hex(seeds[i], factors[i]) for i in range(7)]


def usage_suggestions(palette: List[str], element: str) -> Dict[str, str]:
    return {
        "walls": palette[1],
        "floor": palette[3],
        "accent": palette[0],
        "furniture": palette[4],
        "lighting": palette[2],
    }


def element_hint(element: str) -> str:
    return {
        "water": "흐름·차분·지성의 기운을 살리는 청록·네이비 계열",
        "wood": "성장·생기·확장의 기운을 살리는 그린·라임 계열",
        "fire": "열정·활력·표현의 기운을 살리는 레드·오렌지 계열",
        "earth": "안정·균형·신뢰의 기운을 살리는 샌드·오커 계열",
        "metal": "정제·집중·정확의 기운을 살리는 실버·블루그레이 계열",
    }[element]


# -------------------------------------------------
# /evaluate
# -------------------------------------------------
@app.post("/evaluate", response_model=EvaluateOutput)
async def evaluate(payload: EvaluateInput, request: Request, background_tasks: BackgroundTasks):
    f = payload.features
    s_dir = score_direction_south(f.direction_south)
    s_riv = score_river_distance(f.river_distance)
    s_road = score_road_distance(f.road_distance)
    s_ang = score_angle_diff(f.angle_diff_to_aspect)

    raw = s_dir + s_riv + s_road + s_ang
    total = max(0, min(100, round(raw)))
    element = element_from_score(total)
    msg = message_from_inputs(total, f)

    rec = _build_log_record(
        path="/evaluate",
        request=request,
        payload=payload.model_dump(),
        lat=payload.lat,
        lon=payload.lon,
        score=total,
        element=element,
        details={"direction": s_dir, "river": s_riv, "road": s_road, "angle": s_ang},
    )
    background_tasks.add_task(_insert_log, rec)

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


# -------------------------------------------------
# /color-suggestion
# -------------------------------------------------
@app.post("/color-suggestion", response_model=ColorSuggestOutput)
def color_suggestion(payload: ColorSuggestInput):
    pal = build_palette(payload.element, payload.style or "modern")
    sug = usage_suggestions(pal, payload.element)
    msg = f"{payload.element} 요소에 맞춘 팔레트입니다. {element_hint(payload.element)} · 스타일: {payload.style}."
    return ColorSuggestOutput(
        ok=True,
        element=payload.element,
        style=payload.style or "modern",
        palette=pal,
        suggestions=sug,
        message=msg,
    )


# -------------------------------------------------
# /report  (간단 버전: lat/lng/address만 받아서 특징 생성)
# -------------------------------------------------
class ReportRequest(BaseModel):
    lat: Optional[float] = None
    lng: Optional[float] = None
    address: Optional[str] = None


def _frac(x: float) -> float:
    return abs(x - math.floor(x))


def make_features_from_latlng(lat: float, lng: float) -> dict:
    delta_elevation = abs(_frac(lat) - _frac(lng)) * 100
    river_distance = (_frac(lat * lng) * 1500) + 50
    road_direct = 1.0 if (int(lat * 1000) % 7 == 0) else 0.0
    road_curve = _frac(lat + lng)
    park_count = int((_frac(lat * 37) + _frac(lng * 13)) * 4)
    school_count = int((_frac(lat * 11) + _frac(lng * 17)) * 3)
    industry_distance = 300 + _frac(lat * lng * 3) * 3000
    air_quality = 20 + _frac(lat * 5 + lng * 3) * 60
    noise_level = 40 + (1 - road_curve) * 30 + road_direct * 10
    flood_risk = _frac(lat * 2 - lng * 2) * 100
    return {
        "delta_elevation": round(delta_elevation, 1),
        "river_distance": round(river_distance, 1),
        "road_direct": road_direct,
        "road_curve": round(road_curve, 3),
        "park_count": float(park_count),
        "school_count": float(school_count),
        "industry_distance": round(industry_distance, 1),
        "air_quality": round(air_quality, 1),
        "noise_level": round(noise_level, 1),
        "flood_risk": round(flood_risk, 1),
    }


@app.post("/report")
def simple_report(payload: ReportRequest):
    # 기본 좌표(서울)
    lat = payload.lat or 37.5665
    lng = payload.lng or 126.9780

    feats = make_features_from_latlng(lat, lng)

    recs: List[str] = []
    if feats["noise_level"] > 60:
        recs.append("소음 차단 보강을 권장합니다.")
    if feats["flood_risk"] > 50:
        recs.append("침수/배수 대비 점검이 필요합니다.")
    if not recs:
        recs.append("현재 특별한 리스크는 높지 않습니다.")

    score_total = round(
        85
        - feats["noise_level"] * 0.12
        - feats["flood_risk"] * 0.1
        + feats["park_count"] * 0.8
        + feats["school_count"] * 0.5,
        1,
    )

    return {
        "target": {
            "address": payload.address,
            "lat": lat,
            "lng": lng,
        },
        "features": feats,
        "score": {
            "total": score_total,
            "noise": feats["noise_level"],
            "flood": feats["flood_risk"],
        },
        "recommendations": recs,
    }


# -------------------------------------------------
# /fengshui-score  (간단 점수만)
# -------------------------------------------------
@app.post("/fengshui-score", response_model=ScoreOutput)
def fengshui_score(payload: EvaluateInput):
    f = payload.features
    s_dir = score_direction_south(f.direction_south)
    s_riv = score_river_distance(f.river_distance)
    s_road = score_road_distance(f.road_distance)
    s_ang = score_angle_diff(f.angle_diff_to_aspect)
    raw = s_dir + s_riv + s_road + s_ang
    total = max(0, min(100, round(raw)))
    element = element_from_score(total)
    return ScoreOutput(ok=True, score=total, element=element)
