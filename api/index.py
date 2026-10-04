from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# 1. Resolve project root path robustly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
model_path = os.path.join(BASE_DIR, 'model.pkl')

# Fallback: check relative working directory if BASE_DIR fails
if not os.path.exists(scaler_path):
    scaler_path = 'scaler.pkl'
if not os.path.exists(model_path):
    model_path = 'model.pkl'

# Load pickle files
scaler = pickle.load(open(scaler_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        input_df = pd.DataFrame([{
            'N': float(data['N']),
            'P': float(data['P']),
            'K': float(data['K']),
            'temperature': float(data['temperature']),
            'humidity': float(data['humidity']),
            'ph': float(data['ph']),
            'rainfall': float(data['rainfall'])
        }])
        
        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)[0]
        
        return jsonify({'recommended_crop': str(prediction)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400