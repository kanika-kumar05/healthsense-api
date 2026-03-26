import os
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and preprocessing tools
model = joblib.load("healthsense_model_final.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return "HealthSense API is Running. Use the /predict endpoint via POST."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request (Form or JSON)
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Convert to DataFrame to match training format
        input_df = pd.DataFrame([{
            "heart_rate": float(data["heart_rate"]),
            "spo2": float(data["spo2"]),
            "sleep_hours": float(data["sleep_hours"]),
            "stress_level": float(data["stress_level"]),
            "body_temp": float(data["body_temp"]),
            "age": float(data["age"]),
            "systolic_bp": float(data["systolic_bp"]),
            "diastolic_bp": float(data["diastolic_bp"])
        }])

        # Scale the input
        scaled_data = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_data)
        result = label_encoder.inverse_transform(prediction)[0]

        return jsonify({
            "status": "success",
            "risk_category": result
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)