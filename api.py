"""
api.py  —  Flask backend for AutoPID frontend
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, json
import numpy as np
from ML import predict_optimal_pid

app = Flask(__name__)
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    car_params = request.get_json(force=True)
    required = ['mass','max_acceleration','yaw_inertia','length',
                'max_steering_stiffness_front','max_steering_stiffness_rear',
                'tyre_friction','cd','cross_sectional_area']
    missing = [k for k in required if k not in car_params]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400
    try:
        result = predict_optimal_pid(
            car_params=car_params,
            steer_model_path=os.path.join(BASE_DIR, 'direct_steer_pid_model.pkl'),
            speed_model_path=os.path.join(BASE_DIR, 'direct_speed_pid_model.pkl'),
        )
    except FileNotFoundError as e:
        return jsonify({'error': str(e) + ' — run MLPID_main.py first'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Confidence score
    ranges = {
        'mass':(800,4500),'max_acceleration':(0.8,9.0),'yaw_inertia':(500,12000),
        'length':(2.2,4.8),'max_steering_stiffness_front':(10000,100000),
        'max_steering_stiffness_rear':(10000,100000),'tyre_friction':(0.3,1.4),
        'cd':(0.18,0.55),'cross_sectional_area':(1.6,3.8),
    }
    devs = [abs((car_params[k]-lo)/(hi-lo)-0.5) for k,(lo,hi) in ranges.items()]
    confidence = int(max(50, min(97, round((1 - np.mean(devs)*1.5)*100))))

    return jsonify({**result, 'confidence': confidence})

if __name__ == '__main__':
    print('\n  AutoPID API — open http://localhost:5000\n')
    app.run(debug=True, port=5000)