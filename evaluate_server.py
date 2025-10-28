from flask import Flask, request, jsonify
from flask_cors import CORS
# --- NEW: Pydantic schema (입/출력 계약) ---
from pydantic import BaseModel, Field, ValidationError, conlist, confloat
from typing import Optional, Dict, Any
class Features(BaseModel):
    # 사용 중인 특징만 먼저 선언 (필요하면 추가 확장)
    direction_south: Optional[confloat(ge=0, le=1)] = 0
    river_distance: Optional[confloat(ge=0)] = None
    road_distance: Optional[confloat(ge=0)] = None
    angle_diff_to_aspect: Optional[confloat(ge=0, le=180)] = None

class EvaluateRequest(BaseModel):
    lat: confloat(ge=-90, le=90)
    lon: confloat(ge=-180, le=180)
    features: Features = Field(default_factory=Features)
    rules_version: Optional[str] = "v1"   # 추후 Supabase 룰 버전 스위치용

class EvaluateResponse(BaseModel):
    total: confloat(ge=0, le=100)
    grade: str
    remedies: conlist(str, min_length=0)

def clamp_score(x: float) -> float:
    return max(0.0, min(100.0, x))

app = Flask(__name__)
CORS(app)
@app.get("/")
def root():
    return {"ok": True, "service": "fengshui-api"}, 200

@app.get("/health")
def health():
    return {"status": "ok"}, 200
    
def calc_fengshui_score(payload: dict) -> dict:
    lat = float(payload.get("lat", 0))
    lon = float(payload.get("lon", 0))
    f   = payload.get("features", {}) or {}

    aspect = float(f.get("aspect_deg", 0))
    road   = f.get("road", {}) or {}
    road_dist = float(road.get("distance_m", 0))
    road_angle_diff = float(road.get("angle_diff_to_aspect", 0))

    aspect_score = max(0, 100 - min(abs(aspect - 180), 180))
    road_penalty = max(0, 50 - road_dist * 0.5) + max(0, 30 - (30 - min(road_angle_diff, 30)))
    total = aspect_score - road_penalty
    total = max(0, min(100, total))

    return {
        "subscores": {
            "aspect_score": round(aspect_score, 1),
            "road_penalty": round(road_penalty, 1),
        },
        "total": round(total, 1),
        "grade": "A" if total >= 85 else "B" if total >= 70 else "C" if total >= 55 else "D",
        "remedies": [
            "도로와의 완충녹지 또는 담장 고려",
            "현관/창 배치로 도로 시선 최소화"
        ],
    }

@app.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        payload = request.get_json(silent=True) or {}

        # ✅ Pydantic으로 입력 검증
        req = EvaluateRequest.model_validate(payload)

        # 계산을 위한 데이터 정리
        data = {
            "lat": req.lat,
            "lon": req.lon,
            "features": req.features.model_dump()
        }

        # 실제 점수 계산 (기존 함수 그대로 사용)
        result = calc_fengshui_score(data)

        # 안전 가드 + 표준 응답 스키마
        total = clamp_score(float(result.get("total", 0)))
        grade = result.get("grade", "N/A")
        remedies = result.get("remedies", [])

        resp = EvaluateResponse(total=total, grade=grade, remedies=remedies)
        return jsonify(resp.model_dump()), 200

    except ValidationError as ve:
        return jsonify({"error": "invalid_request", "detail": ve.errors()}), 400
    except Exception as e:
        return jsonify({"error": "internal_error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)






