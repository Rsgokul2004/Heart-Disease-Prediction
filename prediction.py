"""Train the heart-disease model and save it.

The scaler and the classifier are bundled in ONE sklearn Pipeline, so the web
app applies exactly the same preprocessing that was used during training.
Run:  python prediction.py
"""
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = "heart_cleveland_upload.csv"
MODEL_FILE = "heart-disease-model.pkl"
METRICS_FILE = "model_metrics.json"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

heart = pd.read_csv(DATA_FILE).rename(columns={"condition": "target"})
X = heart[FEATURES]
y = heart["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
])

# Honest estimate: 5-fold cross-validation on the training data only
cv = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation accuracy: {cv.mean():.2%} (+/- {cv.std():.2%})")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Hold-out test accuracy: {acc:.2%}\n")
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, MODEL_FILE)
with open(METRICS_FILE, "w") as f:
    json.dump({"test_accuracy": round(acc * 100, 2),
               "cv_accuracy": round(cv.mean() * 100, 2)}, f)
print(f"\nSaved {MODEL_FILE} and {METRICS_FILE}")
