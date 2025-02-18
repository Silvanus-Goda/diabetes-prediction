from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
app = Flask(__name__)

# Load the trained model
model = joblib.load("diabetes_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_values = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(input_values)[0]
        return jsonify({"prediction": int(prediction), "status": "Diabetic" if prediction == 1 else "Non-Diabetic"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the PORT assigned by Render
    app.run(host="0.0.0.0", port=port)
