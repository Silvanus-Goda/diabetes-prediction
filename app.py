from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Replace with your Render API URL
API_URL = "https://diabetes-model-api.onrender.com/predict"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        input_values = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]
        response = requests.post(API_URL, json={"features": input_values})
        prediction = response.json()
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
