from flask import Flask, render_template, request, jsonify
from pathlib import Path
import numpy as np
import joblib

from preprocessing import clean_text


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

app = Flask(__name__)

MODEL_DIR = Path("models")


# ============================================================
# MODEL FILES
# ============================================================

MODEL_PATH = (
    MODEL_DIR /
    "ga_v2_optimized_svm.joblib"
)

VECTORIZER_PATH = (
    MODEL_DIR /
    "tfidf_vectorizer.joblib"
)

FEATURES_PATH = (
    MODEL_DIR /
    "ga_v2_selected_feature_indices.npy"
)


# ============================================================
# LOAD MODEL COMPONENTS
# ============================================================

print("=" * 70)
print("SuicideWatchAI - Application Startup")
print("=" * 70)

print("\nLoading GA V2 model...")

model = joblib.load(
    MODEL_PATH
)

print("GA V2 model loaded successfully.")

print("\nLoading TF-IDF vectorizer...")

vectorizer = joblib.load(
    VECTORIZER_PATH
)

print("TF-IDF vectorizer loaded successfully.")

print("\nLoading GA-selected features...")

selected_features = np.load(
    FEATURES_PATH
)

print(
    f"Selected features loaded: "
    f"{len(selected_features)}"
)

print("\nAll model components loaded successfully.")

print("=" * 70)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# TEXT PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Read JSON request
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({
                "success": False,
                "error": "No input data received."
            }), 400


        text = data.get(
            "text",
            ""
        )


        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if not isinstance(
            text,
            str
        ):

            return jsonify({
                "success": False,
                "error": "Invalid text input."
            }), 400


        if not text.strip():

            return jsonify({
                "success": False,
                "error": "Please enter some text for analysis."
            }), 400


        # ----------------------------------------------------
        # Clean text
        # ----------------------------------------------------

        cleaned_text = clean_text(
            text
        )


        if not cleaned_text:

            return jsonify({
                "success": False,
                "error": "The entered text contains no analyzable content."
            }), 400


        # ----------------------------------------------------
        # TF-IDF transformation
        # ----------------------------------------------------

        tfidf_vector = vectorizer.transform(
            [cleaned_text]
        )


        # ----------------------------------------------------
        # Apply GA V2 feature selection
        # ----------------------------------------------------

        optimized_vector = (
            tfidf_vector[
                :,
                selected_features
            ]
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            optimized_vector
        )[0]


        # ----------------------------------------------------
        # Decision score
        # ----------------------------------------------------

        decision_score = model.decision_function(
            optimized_vector
        )[0]


        # ----------------------------------------------------
        # Convert prediction
        # ----------------------------------------------------

        if prediction == 1:

            result = "Potential Suicide-Ideation Content"

            result_type = "suicide"

        else:

            result = "No Suicide-Ideation Signal Detected"

            result_type = "non-suicide"


        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "prediction": int(
                prediction
            ),

            "result": result,

            "result_type": result_type,

            "decision_score": round(
                float(decision_score),
                4
            )

        })


    except Exception as error:

        print(
            "\nPrediction error:",
            error
        )

        return jsonify({

            "success": False,

            "error":
                "An internal prediction error occurred."

        }), 500


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print(
        "\nStarting SuicideWatchAI web application..."
    )

    print(
        "Open http://127.0.0.1:5000 in your browser."
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )