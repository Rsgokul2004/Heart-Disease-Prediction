# Heart Disease Prediction (Flask)

## Run locally
    python -m venv venv
    venv\Scripts\activate        (Windows)   |   source venv/bin/activate (Linux/Mac)
    pip install -r requirements.txt
    python prediction.py         # trains the model -> heart-disease-model.pkl
    python app.py                # open http://127.0.0.1:5000

Production: `gunicorn app:app` (see Procfile).
The pickled model must be re-created with `python prediction.py` whenever scikit-learn is upgraded.
---

## Project Screenshots

### Home Page
![Home Page](screenshots/home.png)

### Prediction Page
![Prediction Page](screenshots/prediction.png)

### Prediction Result
![Prediction Result](screenshots/result.png)
