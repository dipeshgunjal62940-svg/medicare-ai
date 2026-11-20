import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("Cleaned_Dataset.csv")
print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nDuplicate rows count:", df.duplicated().sum())

# 1. Duplicate rows hatayein
df = df.drop_duplicates().reset_index(drop=True)
print("Shape after deduplication:", df.shape)

# 2. Disease names clean karein
disease_cols = [
    c
    for c in df.columns
    if any(
        k in c.lower()
        for k in ["disease", "prognosis", "condition", "diagnosis"]
    )
]
print("Detected disease columns:", disease_cols)
disease_col = disease_cols[0] if disease_cols else "disease"


def clean_disease_name(val):
  if pd.isna(val):
    return val
  text = str(val).strip()
  text = re.sub(r"\.\.\.+$", "", text).strip()
  if "(" in text and ")" not in text:
    text = text.split("(", 1)[0].strip()

  name_map = {
      "urinary tract infection": "Urinary Tract Infection",
      "urinary tract infection (uti)": "Urinary Tract Infection",
      "uti": "Urinary Tract Infection",
      "myocardial infarction": "Myocardial Infarction",
      "myocardial infarction (heart attack)": "Myocardial Infarction",
      "myocardial infarction (heart": "Myocardial Infarction",
  }
  key = re.sub(r"\s+", " ", text.lower()).strip()
  return name_map.get(key, re.sub(r"\s+", " ", text).strip())


df["disease_clean"] = df[disease_col].apply(clean_disease_name)
print("Unique diseases before cleaning:", df[disease_col].nunique())
print("Unique diseases after cleaning:", df["disease_clean"].nunique())
print("\nAll unique cleaned diseases:\n", sorted(df["disease_clean"].dropna().unique().tolist()))

# 3. outcome_variable aur risk_level vs BP/Cholesterol (Data Leakage Check)
for col in ["outcome_variable", "risk_level"]:
  if col in df.columns:
    print(f"\n--- {col} value counts ---")
    print(df[col].value_counts(dropna=False))

bp_cols = [
    c
    for c in df.columns
    if any(
        k in c.lower()
        for k in ["bp", "blood_pressure", "pressure", "systolic", "diastolic"]
    )
]
chol_cols = [c for c in df.columns if "chol" in c.lower()]
print("\nBP columns:", bp_cols)
print("Cholesterol columns:", chol_cols)

if "risk_level" in df.columns:
  for col in bp_cols + chol_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
      print(f"\n{col} by risk_level:")
      print(
          df.groupby("risk_level", dropna=False)[col].agg(
              ["min", "mean", "max", "count"]
          )
      )
    else:
      print(f"\nCrosstab {col} vs risk_level:")
      print(
          pd.crosstab(
              df[col].fillna("Missing"), df["risk_level"].fillna("Missing")
          )
      )

# 4. Disease distribution
counts = df["disease_clean"].value_counts()
print("\nTop 15 diseases:\n", counts.head(15))
print("Diseases with < 5 rows:", (counts < 5).sum())
print("Diseases with >= 5 rows:", (counts >= 5).sum())

plt.figure(figsize=(10, 6))
top20 = counts.head(20)
sns.barplot(x=top20.values, y=top20.index)
plt.title("Top 20 Diseases After Cleaning")
plt.xlabel("Count")
plt.ylabel("Disease")
plt.tight_layout()
plt.savefig("disease_distribution_top20.png", dpi=150)
plt.show()

df.to_csv("step2_cleaned_eda.csv", index=False)
print("\nSaved step2_cleaned_eda.csv")