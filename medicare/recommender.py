from pathlib import Path
import re
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MED_DB_PATH = BASE_DIR / "medicine_database.pkl"


def load_medicine_db(path: Path = MED_DB_PATH):
  if not path.exists():
    return {}
  data = joblib.load(path)
  if isinstance(data, dict):
    return data
  if isinstance(data, pd.DataFrame):
    # Agar DataFrame ho toh disease -> info dict mein convert karein
    converted = {}
    for _, row in data.iterrows():
      d_key = str(row.get("disease", "")).strip()
      if d_key:
        converted[d_key] = row.to_dict()
    return converted
  return {}


def strip_dosage_info(text: str) -> str:
  """Dosage, frequency ya unit quantity wali lines/phrases hatata hai."""
  if not text:
    return ""
  lines = re.split(r"[\r\n]+", str(text))
  cleaned_lines = []
  dosage_pattern = re.compile(
      r"\b(\d+\s*(mg|mcg|g|ml|tablet(s)?|capsule(s)?|drop(s)?|puff(s)?)|"
      r"dosage|dose|once\s+daily|twice\s+daily|thrice\s+daily|"
      r"every\s+\d+\s+hours?|before\s+meal|after\s+meal)\b",
      re.IGNORECASE,
  )
  for line in lines:
    line = line.strip()
    if not line:
      continue
    if dosage_pattern.search(line):
      continue
    cleaned_lines.append(line)
  return " | ".join(cleaned_lines)


def _format_med_entry(entry) -> dict:
  """Entry ko clean medicine names aur general info mein normalize karta hai."""
  if entry is None:
    return {"medicines": [], "general_info": "information unavailable, doctor se milein"}

  if isinstance(entry, dict):
    raw_meds = (
        entry.get("medicines")
        or entry.get("medicine")
        or entry.get("drugs")
        or []
    )
    if isinstance(raw_meds, str):
      raw_meds = [m.strip() for m in re.split(r",|\n", raw_meds) if m.strip()]
    clean_meds = []
    for m in raw_meds:
      cleaned_m = strip_dosage_info(str(m))
      # Medicine name se trailing dosage numbers trim karein
      cleaned_m = re.sub(
          r"\b\d+\s*(mg|mcg|g|ml)\b.*$", "", cleaned_m, flags=re.IGNORECASE
      ).strip(" -–:,")
      if cleaned_m and cleaned_m not in clean_meds:
        clean_meds.append(cleaned_m)

    raw_info = (
        entry.get("general_info")
        or entry.get("advice")
        or entry.get("description")
        or entry.get("info")
        or ""
    )
    clean_info = strip_dosage_info(str(raw_info))
    if not clean_info:
      clean_info = "General informational guidance only; consult a qualified doctor."

    return {
        "medicines": clean_meds,
        "general_info": clean_info,
    }

  if isinstance(entry, (list, tuple)):
    clean_meds = [
        strip_dosage_info(str(item))
        for item in entry
        if strip_dosage_info(str(item))
    ]
    return {
        "medicines": clean_meds,
        "general_info": "General informational guidance only; consult a qualified doctor.",
    }

  clean_text = strip_dosage_info(str(entry))
  return {
      "medicines": [],
      "general_info": clean_text or "information unavailable, doctor se milein",
  }


def get_disease_recommendation(disease_name: str, med_db: dict) -> dict:
  if not med_db:
    return {
        "available": False,
        "medicines": [],
        "general_info": "information unavailable, doctor se milein",
    }

  # Exact match, phir case-insensitive match
  if disease_name in med_db:
    formatted = _format_med_entry(med_db[disease_name])
    formatted["available"] = True
    return formatted

  lower_map = {str(k).strip().lower(): v for k, v in med_db.items()}
  key = str(disease_name).strip().lower()
  if key in lower_map:
    formatted = _format_med_entry(lower_map[key])
    formatted["available"] = True
    return formatted

  return {
      "available": False,
      "medicines": [],
      "general_info": "information unavailable, doctor se milein",
  }


def build_personalization_warnings(
    age: int,
    blood_pressure,
    cholesterol_level,
    risk_level: str,
    symptoms: dict,
    top_confidence: float,
) -> list[str]:
  warnings = []

  if str(risk_level).strip().lower() == "high":
    warnings.append(
        "High Risk alert: Kripya jaldi se jaldi kisi qualified doctor se salah lein."
    )

  try:
    age_val = int(age)
    if age_val >= 60:
      warnings.append(
          "Age Warning (60+): Senior age group mein symptoms aur vitals par close medical monitoring zaroori hai."
      )
    elif age_val < 18:
      warnings.append(
          "Age Note (<18): Pediatric/adolescent case mein koi bhi medicine ya advice doctor ki dekh-rekh mein hi lein."
      )
  except (TypeError, ValueError):
    pass

  # BP aur Cholesterol numeric (0,1,2) ya string dono handle karein
  bp_str = str(blood_pressure).strip().lower()
  if bp_str in {"2", "high", "hypertension", "stage 2"}:
    warnings.append(
        "Blood Pressure Warning: BP elevated/high level par hai—namak (sodium) kam rakhein aur doctor se BP check karayein."
    )

  chol_str = str(cholesterol_level).strip().lower()
  if chol_str in {"2", "high", "borderline high"}:
    warnings.append(
        "Cholesterol Warning: Cholesterol level high/elevated hai—saturated/trans fats kam karein aur lipid profile par doctor se baat karein."
    )

  # Emergency symptom alert (jaise difficulty_breathing + high fever ya chest/breathing distress)
  breathing = str(symptoms.get("difficulty_breathing", "")).strip().lower() in {"yes", "1", "true"}
  fever = str(symptoms.get("fever", "")).strip().lower() in {"yes", "1", "true"}
  if breathing:
    warnings.append(
        "Emergency Alert: Saans lene mein takleef (difficulty breathing) hone par turant नजदीकी hospital / emergency care se sampark karein."
    )
  elif breathing and fever:
    warnings.append(
        "Urgent Alert: Tez bukhar aur saans ki dikkat ek saath hone par der na karein aur doctor ko dikhayein."
    )

  if top_confidence < 0.25:
    warnings.append(
        f"Low-Confidence Notice: Top prediction confidence ({top_confidence*100:.1f}%) kam hai kyunki dataset chhota aur symptoms limited hain. Ise final diagnosis na maanein."
    )

  return warnings


if __name__ == "__main__":
  db = load_medicine_db()
  print("Medicine DB type:", type(db))
  print("Medicine DB keys:", list(db.keys()) if isinstance(db, dict) else "N/A")
  for test_d in ["Asthma", "Diabetes", "Stroke", "Unknown Disease"]:
    print(f"\n--- {test_d} ---")
    print(get_disease_recommendation(test_d, db))