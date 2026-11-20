import ast
import json
from pathlib import Path
import re
from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np
import pandas as pd
from recommender import build_personalization_warnings, get_disease_recommendation, load_medicine_db

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))  #[cite: 1]

# Load trained models & encoders
risk_model = joblib.load(MODELS_DIR / "risk_model.joblib")
risk_encoder = joblib.load(MODELS_DIR / "risk_encoder.joblib")
disease_model = joblib.load(MODELS_DIR / "disease_model.joblib")
disease_encoder = joblib.load(MODELS_DIR / "disease_encoder.joblib")
feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
med_db = load_medicine_db()  #[cite: 1]

DISCLAIMER = "Yeh sirf information ke liye hai, doctor se salah zaroor lein."


def _parse_general_info_items(raw_info: str) -> list[str]:
  if not raw_info:
    return []
  text = str(raw_info).strip()
  items = []
  if text.startswith("[") and text.endswith("]"):
    try:
      parsed = ast.literal_eval(text)
      if isinstance(parsed, (list, tuple)):
        items = [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
      items = [text]
  else:
    items = [p.strip() for p in re.split(r"\s*\|\s*", text) if p.strip()]

  schedule_re = re.compile(
      r"\b(\d+\s*(mg|mcg|g|ml|min|minutes?|hours?|hrs?)|"
      r"schedule|before\s+meals|after\s+meals|within\s+\d+\s+hours?)\b",
      re.IGNORECASE,
  )
  filtered = [it for it in items if not schedule_re.search(it)]
  return (
      filtered
      if filtered
      else ["General informational guidance only; consult a qualified doctor."]
  )


def _clean_medicine_names(med_list: list[str]) -> list[str]:
  cleaned = []
  for m in med_list:
    s = re.sub(r"\s*-\s*As per .*?schedule.*$", "", str(m), flags=re.IGNORECASE)
    s = re.sub(r"\s*-\s*Hospital only.*$", "", s, flags=re.IGNORECASE).strip()
    if s and s not in cleaned:
      cleaned.append(s)
  return cleaned


def run_prediction_pipeline(payload: dict) -> dict:
  age = int(payload.get("age", 30))
  gender = str(payload.get("gender", "male")).strip().lower()
  if gender not in {"male", "female"}:
    gender = "male"

  bp_raw = payload.get("blood_pressure", 1)
  chol_raw = payload.get("cholesterol_level", 1)

  bp_map = {"low": 0, "normal": 1, "medium": 1, "high": 2, "0": 0, "1": 1, "2": 2}
  chol_map = {
      "low": 0,
      "normal": 1,
      "medium": 1,
      "high": 2,
      "0": 0,
      "1": 1,
      "2": 2,
  }

  blood_pressure = bp_map.get(str(bp_raw).strip().lower(), int(bp_raw))
  cholesterol_level = chol_map.get(str(chol_raw).strip().lower(), int(chol_raw))

  def norm_yn(val):
    s = str(val).strip().lower()
    return "Yes" if s in {"yes", "y", "1", "true", "on"} else "No"

  symptoms = {
      "fever": norm_yn(payload.get("fever", "No")),
      "cough": norm_yn(payload.get("cough", "No")),
      "fatigue": norm_yn(payload.get("fatigue", "No")),
      "difficulty_breathing": norm_yn(
          payload.get("difficulty_breathing", "No")
      ),
  }

  row_dict = {
      "age": age,
      "gender": gender,
      "blood_pressure": blood_pressure,
      "cholesterol_level": cholesterol_level,
      **symptoms,
  }
  X_input = pd.DataFrame([row_dict])[feature_cols]

  # Model A: Risk Level (safe decoding)
  risk_pred_val = risk_model.predict(X_input)[0]
  try:
    risk_pred_int = int(risk_pred_val)
    if 0 <= risk_pred_int < len(risk_encoder.classes_):
      risk_level = str(risk_encoder.classes_[risk_pred_int])
    else:
      risk_level = "Medium"
  except (TypeError, ValueError):
    risk_level = str(risk_pred_val)

  # Model B: Top-3 Diseases
  probs = disease_model.predict_proba(X_input)[0]
  top3_indices = np.argsort(probs)[::-1][:3]

  top_3_diseases = []
  for idx in top3_indices:
    d_name = str(disease_encoder.classes_[idx])
    prob_pct = round(float(probs[idx]) * 100, 2)
    rec = get_disease_recommendation(d_name, med_db)
    rec_clean = {
        "available": rec["available"],
        "medicines": _clean_medicine_names(rec["medicines"]),
        "general_info": (
            [rec["general_info"]]
            if not rec["available"]
            else _parse_general_info_items(rec["general_info"])
        ),
    }
    top_3_diseases.append({
        "disease": d_name,
        "probability_pct": prob_pct,
        "recommendation": rec_clean,
    })

  top_conf = float(probs[top3_indices[0]])
  warnings = build_personalization_warnings(
      age=age,
      blood_pressure=blood_pressure,
      cholesterol_level=cholesterol_level,
      risk_level=risk_level,
      symptoms=symptoms,
      top_confidence=top_conf,
  )

  return {
      "input_summary": row_dict,
      "risk_level": risk_level,
      "top_3_diseases": top_3_diseases,
      "primary_recommendation": top_3_diseases[0]["recommendation"],
      "warnings": warnings,
      "low_confidence": top_conf < 0.25,
      "disclaimer": DISCLAIMER,
  }


@app.route("/", methods=["GET"])
def home():
  return render_template("index.html")  #[cite: 1]


@app.route("/about", methods=["GET"])
def about():
  return render_template("about.html")  #[cite: 1]


@app.route("/contact", methods=["GET"])
def contact():
  return render_template("contact.html")  #[cite: 1]


@app.route("/predict", methods=["POST"])
def predict():
  try:
    if request.is_json:
      payload = request.get_json(silent=True) or {}
    elif request.form:
      payload = request.form.to_dict()
    else:
      raw_body = request.get_data(as_text=True)
      payload = json.loads(raw_body) if raw_body.strip() else {}

    result = run_prediction_pipeline(payload)

    wants_json = (
        request.is_json
        or bool(payload and not request.form)
        or (
            "application/json" in request.headers.get("Accept", "")
            and "text/html" not in request.headers.get("Accept", "")
        )
    )
    if wants_json:
      return jsonify(result), 200

    return render_template("index.html", result=result, form_data=payload)  #[cite: 1]
  except Exception as exc:
    return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
  app.run(host="127.0.0.1", port=5000, debug=True)