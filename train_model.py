# ==========================================
# INSURANCE FRAUD DETECTION 
# ==========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------
df = pd.read_csv("insurance_claims.csv")

df.replace("?", np.nan, inplace=True)
df.drop(columns=["policy_number", "_c39"], errors="ignore", inplace=True)

# Date processing
df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"])
df["incident_date"] = pd.to_datetime(df["incident_date"])

df["policy_year"] = df["policy_bind_date"].dt.year
df["incident_year"] = df["incident_date"].dt.year

df.drop(columns=["policy_bind_date", "incident_date"], inplace=True)

# Fill missing values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        mode_values = df[col].mode(dropna=True)
        fill_value = mode_values.iloc[0] if not mode_values.empty else "unknown"
        df[col] = df[col].fillna(fill_value)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=["object", "string"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ------------------------------------------
# 2. MODEL COMPARISON 
# ------------------------------------------
X = df.drop("fraud_reported", axis=1)
y = df["fraud_reported"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

results = {}
trained_models = {}
for name, model in models.items():
    # Use the same scaled data for all algorithms to keep comparison fair.
    model.fit(X_train_scaled, y_train_smote)
    y_pred = model.predict(X_test_scaled)
    trained_models[name] = model

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {"Accuracy": acc, "F1 Score": f1}

    print(f"\n===== {name} =====")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print(classification_report(y_test, y_pred))

results_df = pd.DataFrame(results).T
try:
    results_df.to_excel("model_comparison.xlsx")
    print("\nModel comparison saved as model_comparison.xlsx")
except Exception:
    # Fallback when optional Excel dependency is missing.
    results_df.to_csv("model_comparison.csv")
    print("\nopenpyxl not found, saved model_comparison.csv instead")
best_model_name = max(results, key=lambda m: results[m]["F1 Score"])
print(f"Best model from full-feature comparison: {best_model_name}")

# ==========================================
# 3. FINAL DEPLOYMENT MODEL (8 FEATURES ONLY)
# ==========================================
print("\n===== Creating Final Deployment Model (8 Features) =====")

feature_columns = [
    "months_as_customer",
    "age",
    "policy_annual_premium",
    "umbrella_limit",
    "total_claim_amount",
    "injury_claim",
    "property_claim",
    "vehicle_claim"
]

feature_ranges = {}
for col in feature_columns:
    numeric_series = pd.to_numeric(df[col], errors="coerce").dropna()
    feature_ranges[col] = {
        "min": float(numeric_series.min()),
        "max": float(numeric_series.max())
    }

X_final = df[feature_columns]
y_final = df["fraud_reported"]
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

# Handle imbalance on training split only.
smote_final = SMOTE(random_state=42)
Xf_train_smote, yf_train_smote = smote_final.fit_resample(Xf_train, yf_train)

# Scale features used for final deployment model.
scaler_final = StandardScaler()
Xf_train_scaled = scaler_final.fit_transform(Xf_train_smote)
Xf_test_scaled = scaler_final.transform(Xf_test)

final_candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

final_scores = {}
for name, candidate in final_candidates.items():
    candidate.fit(Xf_train_scaled, yf_train_smote)
    pred = candidate.predict(Xf_test_scaled)
    final_scores[name] = f1_score(yf_test, pred)

best_final_model_name = max(final_scores, key=final_scores.get)
final_model = final_candidates[best_final_model_name]
print(f"Selected deployment model: {best_final_model_name} (F1={final_scores[best_final_model_name]:.4f})")

# Derive data-driven risk bands from held-out probabilities to avoid
# most predictions collapsing into "Medium Risk".
classes = list(getattr(final_model, "classes_", []))
if 1 in classes:
    pos_idx = classes.index(1)
else:
    pos_idx = 1 if len(classes) > 1 else 0

test_probabilities = final_model.predict_proba(Xf_test_scaled)[:, pos_idx] * 100.0
low_upper = float(np.percentile(test_probabilities, 33))
high_lower = float(np.percentile(test_probabilities, 66))
if high_lower <= low_upper:
    low_upper, high_lower = 40.0, 70.0

# ------------------------------------------
# 4. SAVE DEPLOYMENT PIPELINE
# ------------------------------------------
joblib.dump({
    "model": final_model,
    "scaler": scaler_final,
    "feature_columns": feature_columns,
    "feature_ranges": feature_ranges,
    "selected_model_name": best_final_model_name,
    "risk_thresholds": {
        "low_upper": round(low_upper, 2),
        "high_lower": round(high_lower, 2)
    }
}, "fraud_pipeline.pkl")

print("\nFinal deployment model saved as fraud_pipeline.pkl")
print(f"Risk thresholds -> low < {low_upper:.2f}, high >= {high_lower:.2f}")
print("Ready for Flask Dashboard")