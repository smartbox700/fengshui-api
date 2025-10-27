from flask import Flask, request, jsonify
from flask_cors import CORS

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
    data = request.get_json(silent=True) or {}
    try:
        result = calc_fengshui_score(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "evaluate server running"}), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)


