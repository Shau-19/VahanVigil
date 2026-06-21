"""
Bengaluru Parking Intelligence — Backend API
==============================================
Run:
    pip install fastapi uvicorn pandas numpy
    uvicorn main:app --reload --port 8000
    Then open http://localhost:8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="Bengaluru Parking Intelligence API", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_PATH   = os.path.join(os.path.dirname(__file__), "cluster_priority_final.csv")
STATIC_DIR  = os.path.join(os.path.dirname(__file__), "static")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "isolation_forest_model.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "isolation_forest_scaler.joblib")
_df_cache   = None
_model      = None
_scaler     = None

# Feature order must match exactly how the model was trained in
# notebooks/04_isolation_forest.ipynb -- changing this order without
# retraining will silently produce wrong scores.
ANOMALY_FEATURES = [
    "violations_count", "avg_severity", "junction_pct",
    "avg_vehicle_weight", "multi_violation_pct", "violations_per_hour_std",
]


def load_model():
    """
    Lazy-loads the trained Isolation Forest + scaler saved in Stage 4.
    Returns (None, None) if the files aren't present rather than crashing
    the whole app -- the rest of the API works fine without this model,
    only the /api/score-day endpoint depends on it.
    """
    global _model, _scaler
    if _model is None and os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler


def load_data() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        df = pd.read_csv(DATA_PATH)
        df = df.replace({np.nan: None})
        _df_cache = df
    return _df_cache


def tier_from_priority(score: float) -> str:
    if score >= 80:   return "critical"
    if score >= 20:   return "high"
    if score >= 8:    return "moderate"
    return "low"


def serialize_cluster(row: dict) -> dict:
    def f(v, n=2):   return round(v, n) if v is not None else None
    def pct(v, n=1): return round(v * 100, n) if v is not None else None

    return {
        "cluster_id": int(row["cluster_id"]),
        "station":    row["top_station"],
        "lat":        float(row["centroid_lat"]),
        "lng":        float(row["centroid_lon"]),
        "size":       int(row["size"]),
        "hotspot_type": row["hotspot_type"],
        "tier":       tier_from_priority(row["final_priority_score"]),

        "scores": {
            "final_priority":      f(row["final_priority_score"]),
            "priority_pre_fusion": f(row["priority_score"]),
            "impact_index":        f(row["congestion_impact_index"]),
            "persistence_pct":     pct(row["persistence"]),
            "anomaly_rate_pct":    pct(row["anomaly_rate"]),
        },

        "temporal": {
            "peak_hours":        row["peak_hours"],
            "peak_days":         row["peak_days"],
            "time_risk":         f(row["time_risk_corrected"] * 100 if row.get("time_risk_corrected") is not None else None, 1),
            "night_violation_pct": pct(row.get("night_violation_pct")),
        },

        "coverage": {
            "enforcement_coverage_pct": pct(row["enforcement_coverage_score"]),
            "blind_spot_risk":          f(row["blind_spot_risk"] * 100 if row.get("blind_spot_risk") is not None else None, 1),
            "gap_length_hours":         row["gap_length_hours"],
            "has_blind_spot_flag":      bool((row.get("blind_spot_risk") or 0) > 0.20),
            "has_anomaly_flag":         bool((row.get("anomaly_rate") or 0) > 0.15),
        },

        "ranks": {
            "naive_rank":    int(row["naive_rank"])    if row.get("naive_rank")    is not None else None,
            "impact_rank":   int(row["impact_rank"])   if row.get("impact_rank")   is not None else None,
            "priority_rank": int(row["priority_rank"]) if row.get("priority_rank") is not None else None,
        },

        "road_context": {
            "dominant_road_type": row.get("dominant_road_type"),
            "road_importance":    f(row.get("road_importance")),
        },
    }


@app.get("/api/hotspots")
def get_hotspots():
    df = load_data().sort_values("final_priority_score", ascending=False)
    return {"count": len(df), "hotspots": [serialize_cluster(r) for r in df.to_dict("records")]}


@app.get("/api/hotspots/{cluster_id}")
def get_hotspot(cluster_id: int):
    df = load_data()
    match = df[df["cluster_id"] == cluster_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
    return serialize_cluster(match.to_dict("records")[0])


@app.get("/api/summary")
def get_summary():
    df = load_data()
    critical  = (df["final_priority_score"] >= 80).sum()
    high      = ((df["final_priority_score"] >= 20) & (df["final_priority_score"] < 80)).sum()
    anomalous = (df["anomaly_rate"] > 0.15).sum()
    blind     = (df["blind_spot_risk"] > 0.20).sum()
    top_idx   = df["final_priority_score"].idxmax()
    return {
        "total_violations_analyzed": 115400,
        "total_hotspots_detected":   len(df),
        "critical_priority_zones":   int(critical),
        "high_priority_zones":       int(high),
        "zones_with_anomaly_flag":   int(anomalous),
        "zones_with_blind_spot_flag":int(blind),
        "top_zone": {
            "station":    df.loc[top_idx, "top_station"],
            "cluster_id": int(df.loc[top_idx, "cluster_id"]),
            "score":      round(df.loc[top_idx, "final_priority_score"], 2),
        },
        "data_window": "November 2023 – March 2024",
        "data_note": (
            "Scores are structured estimates (AHP-weighted proxy index), "
            "not direct traffic-flow measurements. Validated for internal "
            "consistency via sensitivity analysis."
        ),
    }


@app.get("/api/ranking")
def get_ranking(limit: int = 30):
    df = load_data().sort_values("final_priority_score", ascending=False).head(limit)
    return [
        {"rank": i+1, "cluster_id": int(r["cluster_id"]), "station": r["top_station"],
         "hotspot_type": r["hotspot_type"], "score": round(r["final_priority_score"], 2),
         "tier": tier_from_priority(r["final_priority_score"])}
        for i, r in enumerate(df.to_dict("records"))
    ]


@app.get("/api/baseline-comparison")
def get_baseline():
    df = load_data().copy()
    df["rank_shift"] = df["naive_rank"] - df["priority_rank"]
    df = df.sort_values("rank_shift", key=abs, ascending=False)
    return [
        {"cluster_id": int(r["cluster_id"]), "station": r["top_station"],
         "violation_count": int(r["size"]),
         "naive_rank":    int(r["naive_rank"]),
         "priority_rank": int(r["priority_rank"]),
         "rank_shift":    int(r["rank_shift"]),
         "final_priority_score": round(r["final_priority_score"], 2)}
        for r in df.head(10).to_dict("records")
    ]


# ─────────────────────────────────────────────────────────
# LIVE ANOMALY SCORING
# Reuses the Isolation Forest trained in notebooks/04_isolation_forest.ipynb
# (saved model + scaler) to score a hypothetical day's stats for a cluster
# on demand, without retraining. This is the one live-inference endpoint
# in the API -- everything else in cluster_priority_final.csv was scored
# once during the monthly pipeline run, not per-request.
# ─────────────────────────────────────────────────────────

class DayStats(BaseModel):
    """
    One day's aggregated violation stats for a single cluster -- the same
    feature shape the Isolation Forest was trained on in Stage 4.
    """
    violations_count: int = Field(..., ge=0, example=18,
        description="Total violations recorded that day for this cluster")
    avg_severity: float = Field(..., ge=0, le=4, example=2.4,
        description="Average severity score (0-4 scale) for that day's violations")
    junction_pct: float = Field(..., ge=0, le=1, example=0.55,
        description="Fraction of that day's violations occurring at a junction")
    avg_vehicle_weight: float = Field(..., ge=0, example=1.1,
        description="Average congestion weight of vehicles involved that day")
    multi_violation_pct: float = Field(..., ge=0, le=1, example=0.12,
        description="Fraction of challans with 2+ violations that day")
    violations_per_hour_std: float = Field(..., ge=0, example=1.8,
        description="Standard deviation of violation counts across that day's hours")


@app.post("/api/score-day")
def score_day(stats: DayStats):
    """
    Live inference: scores a hypothetical day's cluster stats against the
    trained Isolation Forest, returning an anomaly score and flag.

    Use case: between monthly pipeline refreshes, an officer or analyst can
    check whether today's observed numbers for a zone look unusual relative
    to the historical pattern the model learned -- without waiting for the
    next full pipeline run.

    Requires isolation_forest_model.joblib and isolation_forest_scaler.joblib
    to be present in the backend/ folder (saved by notebooks/04_isolation_forest.ipynb).
    If they're missing, this endpoint returns a 503 rather than crashing
    the rest of the API -- every other endpoint works fine without them.
    """
    model, scaler = load_model()
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Isolation Forest model not loaded. Copy "
                "isolation_forest_model.joblib and isolation_forest_scaler.joblib "
                "from notebooks/04_isolation_forest.ipynb's output into the "
                "backend/ folder, then restart the server."
            ),
        )

    row = pd.DataFrame([{f: getattr(stats, f) for f in ANOMALY_FEATURES}])
    X_scaled = scaler.transform(row[ANOMALY_FEATURES])

    raw_score   = float(model.decision_function(X_scaled)[0])
    is_anomaly  = bool(model.predict(X_scaled)[0] == -1)

    return {
        "input": stats.dict(),
        "anomaly_score": round(raw_score, 4),
        "is_anomaly": is_anomaly,
        "interpretation": (
            "Unusual -- this day's pattern deviates from what the model learned "
            "as normal for a typical cluster-day." if is_anomaly else
            "Within normal range -- consistent with typical cluster-day patterns "
            "seen in the training data."
        ),
        "note": (
            "Lower anomaly_score = more anomalous. This is a live prediction "
            "from the saved Stage 4 model, computed on-demand for this request "
            "-- it is not stored or reflected in cluster_priority_final.csv "
            "until the next full pipeline run."
        ),
    }


@app.get("/", include_in_schema=False)
def serve_dashboard():
    idx = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(idx):
        raise HTTPException(404, "static/index.html not found")
    return FileResponse(idx)


@app.get("/api", include_in_schema=False)
def api_root():
    return {"service": "Bengaluru Parking Intelligence API",
            "endpoints": ["/api/hotspots", "/api/hotspots/{cluster_id}",
                          "/api/summary", "/api/ranking", "/api/baseline-comparison",
                          "/api/score-day (POST, live Isolation Forest inference)"]}


assets_dir = os.path.join(STATIC_DIR, "assets")
if os.path.isdir(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")