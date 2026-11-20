from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(BASE_DIR / "step3_filtered_dataset.csv")

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

numeric_features = ["age", "blood_pressure", "cholesterol_level"]
categorical_features = [
    "gender",
    "fever",
    "cough",
    "fatigue",
    "difficulty_breathing",
]

X = df[feature_cols].copy()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
    ]
)

candidates = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "SVC": SVC(
        kernel="rbf", probability=True, random_state=42, class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
    "macro_recall": make_scorer(recall_score, average="macro", zero_division=0),
}


def evaluate_task(task_name, y_raw):
  le = LabelEncoder()
  y = le.fit_transform(y_raw)

  print(f"\n{'='*65}")
  print(f"EVALUATING: {task_name} (n_classes={len(le.classes_)})")
  print(f"{'='*65}")

  results = []
  best_score = -1.0
  best_name = None
  best_pipeline = None

  for name, clf in candidates.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    acc = scores["test_accuracy"].mean()
    f1 = scores["test_macro_f1"].mean()
    rec = scores["test_macro_recall"].mean()

    results.append({
        "Model": name,
        "CV Accuracy": round(acc, 4),
        "CV Macro-F1": round(f1, 4),
        "CV Macro-Recall": round(rec, 4),
    })

    # Selection primary metric: Macro-F1 (tie-breaker: Accuracy)
    selection_metric = (f1, acc)
    if selection_metric > (best_score, -1.0):
      best_score = f1
      best_name = name
      best_pipeline = pipe

  res_df = pd.DataFrame(results).sort_values(
      by=["CV Macro-F1", "CV Accuracy"], ascending=False
  )
  print(res_df.to_string(index=False))

  # Fit best pipeline on full filtered dataset for deployment
  best_pipeline.fit(X, y)
  print(f"\nBest model for {task_name}: {best_name}")
  return best_pipeline, le, res_df


# Model A: Risk Level
risk_model, risk_encoder, risk_results = evaluate_task(
    "Model A - Risk Level", df["risk_level"]
)

# Model B: Disease Prediction
disease_model, disease_encoder, disease_results = evaluate_task(
    "Model B - Disease Prediction", df["disease_final"]
)

# Save artifacts into models/
joblib.dump(risk_model, MODELS_DIR / "risk_model.joblib")
joblib.dump(risk_encoder, MODELS_DIR / "risk_encoder.joblib")
joblib.dump(disease_model, MODELS_DIR / "disease_model.joblib")
joblib.dump(disease_encoder, MODELS_DIR / "disease_encoder.joblib")
joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")

print("\nSaved trained models & encoders in 'models/' folder successfully.")

# Quick Top-3 sample test on first row
sample_probs = disease_model.predict_proba(X.iloc[[0]])[0]
top3_idx = np.argsort(sample_probs)[::-1][:3]
top3_diseases = [
    (disease_encoder.classes_[i], round(float(sample_probs[i]) * 100, 2))
    for i in top3_idx
]
print("\nSample Top-3 Disease Prediction (Row 0):", top3_diseases)