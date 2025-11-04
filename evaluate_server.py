from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Dict
from typing import List, Optional
from fastapi import Request, BackgroundTasks
from supabase import create_client, Client  # 이미 있으면 중복 X
import os                                  # 이미 있으면 중복 X

# --- Supabase client (환경변수 필요: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client | None = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

async def _insert_log(record: dict):
    """비동기 백그라운드에서 Supabase에 로그 적재 (실패해도 API 응답에는 영향 X)"""
    try:
        if not supabase:
            return
        supabase.table("api_logs").insert(record).execute()
    except Exception:
        # 로깅 실패는 조용히 무시
        pass

def _build_log_record(
    path: str,
    request: Request,
    payload: dict | None,
    lat: float | None = None,
    lon: float | None = None,
    score: int | None = None,
    element: str | None = None,
    details: dict | None = None,
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


app = FastAPI(
    title="fengshui-api",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
# --- Supabase 연결 설정 ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


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
async def evaluate(payload: EvaluateInput, request: Request, background_tasks: BackgroundTasks):

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
            # --- 로그 저장 ---
    

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
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Color Suggestion (Five Elements) ----------

class ColorSuggestInput(BaseModel):
    element: Literal["water", "wood", "fire", "earth", "metal"]
    style: Optional[Literal["modern", "minimal", "vintage", "lux"]] = "modern"

class ColorSuggestOutput(BaseModel):
    ok: Literal[True]
    element: Literal["water", "wood", "fire", "earth", "metal"]
    style: Literal["modern", "minimal", "vintage", "lux"]
    palette: List[str]                 # HEX 리스트 (밝은톤 → 진한톤)
    suggestions: Dict[str, str]        # 용도별 추천 HEX
    message: str

BASE_PALETTES: Dict[str, List[str]] = {
    "water": ["#0EA5E9", "#38BDF8", "#93C5FD", "#111827"],
    "wood":  ["#16A34A", "#86EFAC", "#A3E635", "#374151"],
    "fire":  ["#EF4444", "#F97316", "#FCA5A5", "#1F2937"],
    "earth": ["#A16207", "#D6A354", "#F5E6CC", "#4B5563"],
    "metal": ["#94A3B8", "#CBD5E1", "#E5E7EB", "#0F172A"],
}

def clamp(v: int) -> int:
    return max(0, min(255, v))

def adjust_hex(hex_color: str, factor: float) -> str:
    """HEX 색상을 밝게(>1.0) 또는 어둡게(<1.0) 조정."""
    hex_color = hex_color.lstrip("#")
    r = clamp(int(int(hex_color[0:2], 16) * factor))
    g = clamp(int(int(hex_color[2:4], 16) * factor))
    b = clamp(int(int(hex_color[4:6], 16) * factor))
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def build_palette(element: str, style: str) -> List[str]:
    base = BASE_PALETTES[element]
    if style == "minimal":
        factors = [1.25, 1.05, 0.9, 0.7, 0.5, 0.35, 0.2]
    elif style == "vintage":
        factors = [0.9, 0.8, 0.7, 1.0, 1.1, 0.6, 0.45]
    elif style == "lux":
        factors = [0.85, 0.7, 0.55, 1.0, 1.15, 0.4, 0.25]
    else:  # modern
        factors = [1.15, 1.0, 0.85, 0.7, 0.55, 0.4, 0.25]

    c0, c1, c2, cN = base[0], base[1], base[2], base[3]
    seeds = [c0, c1, c2, cN, c0, c1, cN]
    return [adjust_hex(seeds[i], factors[i]) for i in range(7)]

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
        "wood":  "성장·생기·확장의 기운을 살리는 그린·라임 계열",
        "fire":  "열정·활력·표현의 기운을 살리는 레드·오렌지 계열",
        "earth": "안정·균형·신뢰의 기운을 살리는 샌드·오커 계열",
        "metal": "정제·집중·정확의 기운을 살리는 실버·블루그레이 계열",
    }[element]

@app.post("/color-suggestion", response_model=ColorSuggestOutput)
def color_suggestion(payload: ColorSuggestInput):
    try:
        pal = build_palette(payload.element, payload.style)
        sug = usage_suggestions(pal, payload.element)
        msg = f"{payload.element} 요소에 맞춘 팔레트입니다. {element_hint(payload.element)} · 스타일: {payload.style}."
        return ColorSuggestOutput(
            ok=True,
            element=payload.element,
            style=payload.style or "modern",
            palette=pal,
            suggestions=sug,
            message=msg
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Report (Evaluate + Color Suggestion) ----------

from typing import List, Optional  # 없으면 추가

class ReportInput(EvaluateInput):
    # 색상 스타일 선택(선택)
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

def recommendations_from_inputs(f: FeatureData) -> List[str]:
    tips: List[str] = []
    # 방위
    if f.direction_south <= 0.3:
        tips.append("북향 성향이 강하므로 따뜻한 조명과 밝은 벽면 색으로 채광을 보완하세요.")
    elif f.direction_south < 0.7:
        tips.append("남동·남서 방향 가구 배치로 채광과 통풍을 확보하세요.")
    else:
        tips.append("남향 장점을 살려 거실·작업공간을 창 쪽으로 배치하세요.")
    # 하천
    if f.river_distance < 0.1:
        tips.append("하천과 매우 가까우면 제습/방수에 주의하고 창틀 결로 점검을 권장합니다.")
    elif not (0.3 <= f.river_distance <= 0.8):
        tips.append("하천과의 거리가 멀면 수기(水氣) 보완을 위해 청록·네이비 소품을 포인트로 사용하세요.")
    # 도로
    if f.road_distance < 0.05:
        tips.append("주요 도로 인접 시 방음커튼·러그로 소음·진동을 줄이세요.")
    elif f.road_distance > 1.2:
        tips.append("도로와 거리가 멀면 접근성이 떨어질 수 있으니 동선 계획을 최적화하세요.")
    # 각도
    if f.angle_diff_to_aspect >= 45:
        tips.append("방위 편차가 커서 책상·침대 방향을 남·동남으로 재배치하는 것을 권장합니다.")
    elif f.angle_diff_to_aspect > 15:
        tips.append("부분적으로 방위를 보정하기 위해 포인트 벽면을 남향 기준으로 잡아주세요.")

    if not tips:
        tips.append("현재 입지는 균형이 좋아 기본 동선만 정리해도 충분합니다.")
    return tips

@app.post("/report", response_model=ReportOutput)
def report(payload: ReportInput):
    """
    Evaluate + Color Suggestion을 합친 종합 리포트.
    입력: EvaluateInput + style
    출력: 점수, 오행, 요약, 보완 팁, 팔레트, 용도별 색상, 디테일 점수
    """
    try:
        f = payload.features
        # 1) 점수 계산
        s_dir = score_direction_south(f.direction_south)
        s_riv = score_river_distance(f.river_distance)
        s_road = score_road_distance(f.road_distance)
        s_ang = score_angle_diff(f.angle_diff_to_aspect)
        raw = s_dir + s_riv + s_road + s_ang
        total = max(0, min(100, round(raw)))
        element = element_from_score(total)

        # 2) 요약/추천
        message = message_from_inputs(total, f)
        tips = recommendations_from_inputs(f)

        # 3) 팔레트/용도
        pal = build_palette(element, payload.style or "modern")
        sug = usage_suggestions(pal, element)

        # 4) 응답
        return ReportOutput(
            ok=True,
            score=total,
            element=element,
            summary=message,
            recommendations=tips,
            palette=pal,
            suggestions=sug,
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

# ---------- Report (Evaluate + Color Suggestion) ----------
from typing import List, Optional  # 없으면 유지/추가

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

def recommendations_from_inputs(f: FeatureData) -> List[str]:
    tips: List[str] = []
    # 방위
    if f.direction_south <= 0.3:
        tips.append("북향 성향이 강하므로 따뜻한 조명과 밝은 벽면 색으로 채광을 보완하세요.")
    elif f.direction_south < 0.7:
        tips.append("남동·남서 방향 가구 배치로 채광과 통풍을 확보하세요.")
    else:
        tips.append("남향 장점을 살려 거실·작업공간을 창 쪽으로 배치하세요.")
    # 하천
    if f.river_distance < 0.1:
        tips.append("하천과 매우 가까우면 제습/방수에 주의하고 창틀 결로 점검을 권장합니다.")
    elif not (0.3 <= f.river_distance <= 0.8):
        tips.append("하천과의 거리가 멀면 수기 보완을 위해 청록·네이비 소품을 포인트로 사용하세요.")
    # 도로
    if f.road_distance < 0.05:
        tips.append("주요 도로 인접 시 방음커튼·러그로 소음·진동을 줄이세요.")
    elif f.road_distance > 1.2:
        tips.append("도로와 거리가 멀면 접근성이 떨어질 수 있으니 동선 계획을 최적화하세요.")
    # 각도
    if f.angle_diff_to_aspect >= 45:
        tips.append("방위 편차가 커서 책상·침대 방향을 남·동남으로 재배치하세요.")
    elif f.angle_diff_to_aspect > 15:
        tips.append("포인트 벽면을 남향 기준으로 잡아 방위를 보정하세요.")
    if not tips:
        tips.append("현재 입지는 균형이 좋아 기본 동선만 정리해도 충분합니다.")
    return tips

@app.post("/report", response_model=ReportOutput)
def report(payload: ReportInput):
    try:
        f = payload.features
        # 1) 점수 계산
        s_dir = score_direction_south(f.direction_south)
        s_riv = score_river_distance(f.river_distance)
        s_road = score_road_distance(f.road_distance)
        s_ang = score_angle_diff(f.angle_diff_to_aspect)
        raw = s_dir + s_riv + s_road + s_ang
        total = max(0, min(100, round(raw)))
        element = element_from_score(total)

        # 2) 요약/추천
        summary = message_from_inputs(total, f)
        tips = recommendations_from_inputs(f)

        # 3) 팔레트/용도
        pal = build_palette(element, payload.style or "modern")
        sug = usage_suggestions(pal, element)

        return ReportOutput(
            ok=True,
            score=total,
            element=element,
            summary=summary,
            recommendations=tips,
            palette=pal,
            suggestions=sug,
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

# ---------- Lightweight Fengshui Score ----------
class ScoreOutput(BaseModel):
    ok: Literal[True]
    score: int
    element: Literal["water", "wood", "fire", "earth", "metal"]

@app.post("/fengshui-score", response_model=ScoreOutput)
def fengshui_score(payload: EvaluateInput):
    """
    경량 점수 API: 점수와 오행만 반환 (리스트/미리보기 용)
    """
    try:
        f = payload.features
        s_dir = score_direction_south(f.direction_south)
        s_riv = score_river_distance(f.river_distance)
        s_road = score_road_distance(f.road_distance)
        s_ang = score_angle_diff(f.angle_diff_to_aspect)
        raw = s_dir + s_riv + s_road + s_ang
        total = max(0, min(100, round(raw)))
        element = element_from_score(total)
        return ScoreOutput(ok=True, score=total, element=element)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
















