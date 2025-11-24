# CODESOFT — Machine Learning Tasks

This repository contains three separate internship task projects. Each task is self-contained in its folder and includes notebooks, models, and minimal instructions to run or inspect the work.

## Tasks Overview

- **Task 1 — Movie Genre Prediction** (`task-1`)
	- Contents: `movie_genre_prediction (2).ipynb`
	- Description: Jupyter notebook that explores and builds a model to predict movie genres (notebook contains dataset analysis and modeling steps).

- **Task 2 — Credit Card Fraud Detection** (`task-2`)
	- Contents: `Credit_Card_Fraud_Detection (1).ipynb`
	- Description: Notebook demonstrating a fraud-detection pipeline using classical ML techniques and evaluation.

- **Task 3 — Spam Detector (Flask app)** (`task-3`)
	- Contents: `app.py`, `Spam_Detection.ipynb`, `Spam_Detection.pkl`, `tfidf_vectorizer.pkl`, `requirements.txt`, `spam_dataset.xlsx`
	- Description: A minimal Flask web application that classifies text messages as spam or not spam using a pre-trained TF-IDF vectorizer and classifier.

## Quick Setup (Task 3)
Task 3 is runnable as a small web app. To set up and run it locally:

```powershell
cd task-3
python -m pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` in a browser to use the provided web UI, or POST JSON to `/predict`:

```http
POST /predict
Content-Type: application/json

{"message": "your message here"}
```

## Files (top-level)
- `task-1/` — Movie genre prediction notebook
- `task-2/` — Credit card fraud detection notebook
- `task-3/` — Spam detector Flask app and trained artifacts
- `templates/` — HTML template used by the Flask app
- `README.md` — this file

## Notes
- Keep the model artifacts (`*.pkl`) inside their respective task folders (they are included in `task-3`).
- The notebooks are intended for review; run cells in an environment with the listed dependencies.

---

This README contains only project-related information for submission.
