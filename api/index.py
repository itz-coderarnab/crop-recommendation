from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app) # Enables cross-origin requests from frontend

# Get the directory where THIS script resides
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Check if model files are in current dir or parent dir
scaler_path = os.path.join(CURRENT_DIR, 'scaler.pkl')
if not os.path.exists(scaler_path):
    scaler_path = os.path.join(os.path.dirname(CURRENT_DIR), 'scaler.pkl')

model_path = os.path.join(CURRENT_DIR, 'model.pkl')
if not os.path.exists(model_path):
    model_path = os.path.join(os.path.dirname(CURRENT_DIR), 'model.pkl')

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

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)