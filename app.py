import json
import os

import joblib
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "heart-disease-model.pkl"))

try:
    with open(os.path.join(BASE_DIR, "model_metrics.json")) as f:
        ACCURACY = json.load(f)["cv_accuracy"]
except (OSError, KeyError, ValueError):
    ACCURACY = None

app = Flask(__name__)

THRESHOLD = 0.5  # standard decision boundary

# field -> (type, min, max, label)  - ranges match the training dataset
NUMERIC = {
    "age":      (int,   29,  77,  "Age"),
    "trestbps": (int,   94,  200, "Resting blood pressure"),
    "chol":     (int,   126, 564, "Serum cholesterol"),
    "thalach":  (int,   71,  202, "Max heart rate"),
    "oldpeak":  (float, 0,   6.2, "ST depression"),
    "ca":       (int,   0,   3,   "Number of major vessels"),
}
CHOICES = {
    "sex": {0, 1}, "cp": {0, 1, 2, 3}, "fbs": {0, 1}, "restecg": {0, 1, 2},
    "exang": {0, 1}, "slope": {0, 1, 2}, "thal": {0, 1, 2},
}
CHOICE_LABELS = {
    "sex": "Sex", "cp": "Chest pain type", "fbs": "Fasting blood sugar",
    "restecg": "Resting ECG results", "exang": "Exercise-induced angina",
    "slope": "Slope of peak exercise ST segment", "thal": "Thalassemia",
}
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def parse_form(form):
    """Return (values, errors). Never raises on bad input."""
    values, errors = {}, []
    for name, (cast, lo, hi, label) in NUMERIC.items():
        raw = (form.get(name) or "").strip()
        if not raw:
            errors.append(f"{label} is required.")
            continue
        try:
            val = cast(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue
        if not lo <= val <= hi:
            errors.append(f"{label} must be between {lo} and {hi}.")
            continue
        values[name] = val
    for name, allowed in CHOICES.items():
        raw = (form.get(name) or "").strip()
        try:
            val = int(raw)
        except ValueError:
            errors.append(f"Please select a value for {CHOICE_LABELS[name]}.")
            continue
        if val not in allowed:
            errors.append(f"Invalid value for {CHOICE_LABELS[name]}.")
            continue
        values[name] = val
    return values, errors


@app.route("/")
def home():
    return render_template("main.html", accuracy=ACCURACY, errors=[], form={})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("home"))

    values, errors = parse_form(request.form)
    if errors:
        return render_template("main.html", accuracy=ACCURACY,
                               errors=errors, form=request.form), 400

    data = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    prob = float(model.predict_proba(data)[0][1])
    prediction = 1 if prob >= THRESHOLD else 0
    return render_template("result.html", prediction=prediction,
                           probability=round(prob * 100, 1))


@app.errorhandler(404)
def not_found(_):
    return render_template("main.html", accuracy=ACCURACY,
                           errors=["Page not found."], form={}), 404


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
