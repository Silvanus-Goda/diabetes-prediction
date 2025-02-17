import joblib
import numpy as np
from flask import Flask, request, jsonify

# Load the trained model
model = joblib.load("diabetes_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Diabetes Prediction Model API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(input_data)
        outcome = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        return jsonify({"prediction": outcome})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
