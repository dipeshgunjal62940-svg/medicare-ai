from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "step2_cleaned_eda.csv")

# Bache hue truncated names ko exact normalize karein
extra_map = {
    "chronic obstructive pulmonary": (
        "Chronic Obstructive Pulmonary Disease (COPD)"
    ),
    "obsessive-compulsive disorde": "Obsessive-Compulsive Disorder",
}


def finalize_disease_name(name):
  if pd.isna(name):
    return name
  text = re.sub(r"\s+", " ", str(name).strip())
  return extra_map.get(text.lower(), text)


df["disease_final"] = df["disease_clean"].apply(finalize_disease_name)

# >= 5 rows wali diseases filter karein
counts = df["disease_final"].value_counts()
valid_diseases = counts[counts >= 5].index
df_filtered = df[df["disease_final"].isin(valid_diseases)].copy()

print("Original deduplicated rows:", len(df))
print("Filtered rows (diseases with >= 5 rows):", len(df_filtered))
print("Final number of diseases:", df_filtered["disease_final"].nunique())
print(
    "\nFinal diseases & row counts:\n",
    df_filtered["disease_final"].value_counts(),
)

# Feature types aur unique values check karein (Step 4 ke liye)
feature_cols = [
    "age",
    "gender",
    "blood_pressure",
    "cholesterol_level",
    "fever",
    "cough",
    "fatigue",
    "difficulty_breathing",
]
print("\n--- Feature Inspection ---")
for col in feature_cols:
  if col in df_filtered.columns:
    uniq = df_filtered[col].unique()
    sample = uniq[:10] if len(uniq) > 10 else uniq
    print(f"{col} (dtype={df_filtered[col].dtype}, nunique={len(uniq)}): {sample}")

df_filtered.to_csv(BASE_DIR / "step3_filtered_dataset.csv", index=False)
print("\nSaved 'step3_filtered_dataset.csv' successfully.")