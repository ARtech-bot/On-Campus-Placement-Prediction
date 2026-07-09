"""
train_and_save_model.py

Recreates the preprocessing from your DS_project.ipynb notebook, fixes two
methodology issues (scaler/SMOTE fit on the full dataset instead of the
training split only), trains the same candidate models your notebook tried
(minus XGBoost — its compiled binary alone blows past Vercel's 500MB function
size limit, and it wasn't reliably beating the sklearn models anyway), and
saves whichever scores highest on the held-out test set.

Artifacts written:
    model.pkl              -> best classifier
    scaler.pkl              -> RobustScaler fit on the TRAIN split only
    feature_columns.pkl     -> exact column order the model expects
    label_mappings.json     -> Yes/No -> 1/0 mappings used for categoricals

Usage:
    python train_and_save_model.py
"""

import json

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

DATA_PATH = "placementdata.csv"

# ---- 1. Load & clean (same as your notebook) ----
df = pd.read_csv(DATA_PATH)
df.drop(columns=["StudentID"], inplace=True)
df = df.drop_duplicates()

# ---- 2. Encode categoricals with explicit, deployment-safe mappings ----
BINARY_MAP = {"No": 0, "Yes": 1}
STATUS_MAP = {"NotPlaced": 0, "Placed": 1}

df["ExtracurricularActivities"] = df["ExtracurricularActivities"].map(BINARY_MAP)
df["PlacementTraining"] = df["PlacementTraining"].map(BINARY_MAP)
df["PlacementStatus"] = df["PlacementStatus"].map(STATUS_MAP)

x = df.drop(columns=["PlacementStatus"])
y = df["PlacementStatus"]
feature_columns = list(x.columns)

# ---- 3. Split FIRST, to avoid leaking test data into scaling/resampling ----
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# ---- 4. Scale — fit on train only, apply same transform to test ----
scaler = RobustScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ---- 5. Balance classes — SMOTE on train only ----
smote = SMOTE(random_state=42)
x_train_res, y_train_res = smote.fit_resample(x_train_scaled, y_train)

# ---- 6. Train each candidate model from your notebook, compare on test ----
candidates = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LogisticRegression_ElasticNet": LogisticRegression(
        penalty="elasticnet", l1_ratio=1, solver="saga", max_iter=2000, random_state=100
    ),
    "RandomForest": RandomForestClassifier(random_state=42),
    "RandomForest_gini_sqrt": RandomForestClassifier(
        criterion="gini", max_features="sqrt", random_state=0
    ),
}

results = []
fitted_models = {}
for name, clf in candidates.items():
    clf.fit(x_train_res, y_train_res)
    pred = clf.predict(x_test_scaled)
    acc = accuracy_score(y_test, pred)
    results.append((name, acc))
    fitted_models[name] = clf
    print(f"{name}: accuracy={acc:.4f}")

results.sort(key=lambda r: r[1], reverse=True)
best_name, best_acc = results[0]
best_model = fitted_models[best_name]

print(f"\nBest model: {best_name} (accuracy={best_acc:.4f})")
print(classification_report(y_test, best_model.predict(x_test_scaled)))

# ---- 7. Save artifacts ----
joblib.dump(best_model, "api/model.pkl")
joblib.dump(scaler, "api/scaler.pkl")
joblib.dump(feature_columns, "api/feature_columns.pkl")
with open("api/label_mappings.json", "w") as f:
    json.dump({"binary": BINARY_MAP, "status": STATUS_MAP, "best_model": best_name}, f, indent=2)

print("\nSaved artifacts into api/: model.pkl, scaler.pkl, feature_columns.pkl, label_mappings.json")
print("Commit these files to git, then run 'vercel' to deploy.")