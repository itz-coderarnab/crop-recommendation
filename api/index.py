from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler relative to current directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb'))
model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = np.array([[
            data['N'], data['P'], data['K'],
            data['temperature'], data['humidity'],
            data['ph'], data['rainfall']
        ]])
        
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        
        return jsonify({'recommended_crop': str(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Vercel needs the WSGI app instance exported