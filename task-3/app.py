import os
import sys
try:
    import joblib
except Exception as e:
    print("Missing dependency 'joblib':", e)
    print("Install required packages with: pip install -r requirements.txt")
    sys.exit(1)

try:
    from flask import Flask, request, jsonify, render_template
except Exception as e:
    print("Missing dependency or failed import:", e)
    print("Install required packages with: pip install -r requirements.txt")
    sys.exit(1)


app = Flask(__name__)

# Load lazily so the app can report missing files clearly instead of failing at import time
model = None
vectorizer = None


def load_models():
    """Load model and vectorizer from files next to this script.

    Raises FileNotFoundError with a clear message if files are missing.
    """
    global model, vectorizer
    if model is not None and vectorizer is not None:
        return

    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "Spam_Detection.pkl")
    vec_path = os.path.join(base, "tfidf_vectorizer.pkl")

    missing = []
    if not os.path.exists(model_path):
        missing.append(model_path)
    if not os.path.exists(vec_path):
        missing.append(vec_path)
    if missing:
        raise FileNotFoundError(f"Missing model files: {', '.join(missing)}\nEnsure the files exist and are named correctly.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure models are loaded (gives clear errors if files are missing)
        load_models()

        data = request.get_json(silent=True)
        if data is None:
            # support form POST from the HTML page
            msg = request.form.get("message", "")
        else:
            msg = data.get("message", "")

        if not msg.strip():
            return jsonify({"error": "Empty message."}), 400
        

        vec = vectorizer.transform([msg])
        if vec.nnz == 0:
            return jsonify({"error": "Message has no recognizable features."}), 400

        pred = model.predict(vec)[0]

        # Try to get probability if available, otherwise fall back to decision function or N/A
        confidence = "N/A"
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(vec)[0]
                confidence = f"{max(proba) * 100:.2f}%"
            except Exception:
                confidence = "N/A"
        elif hasattr(model, "decision_function"):
            try:
                score = model.decision_function(vec)[0]
                confidence = f"score={float(score):.4f}"
            except Exception:
                confidence = "N/A"

        return jsonify({
            "prediction": "Spam" if int(pred) == 1 else "Not Spam",
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
