
from flask import Flask, request, render_template
import pickle
import numpy as np
import os

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "Model.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input values from the form
        int_features = [
            int(value) for value in request.form.values()
        ]

        # Prepare features for prediction
        final_features = np.array(int_features).reshape(1, -1)

        # Predict using the trained model
        prediction = model.predict(final_features)

        output = "Placed" if prediction[0] == 1 else "Not Placed"

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {output}"
        )

    except (ValueError, TypeError) as e:
        app.logger.exception("Invalid prediction input")
        return "Invalid input. Please check the entered values.", 400

    except Exception:
        app.logger.exception("Prediction failed")
        return "Prediction failed. Please check the application logs.", 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False
    )