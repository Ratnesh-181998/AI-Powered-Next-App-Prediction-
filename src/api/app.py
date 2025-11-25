"""
Flask API for App Prediction
Provides REST API endpoints for making predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.predictor import AppPredictor

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize predictor
try:
    predictor = AppPredictor('models/xgboost_v1.pkl')
    print("[OK] Model loaded successfully")
except Exception as e:
    print(f"[WARNING] Could not load model: {e}")
    print("   API will run in demo mode")
    predictor = None


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'iPhone App Prediction API',
        'version': '1.0',
        'model_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'uptime': 'N/A',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make app prediction
    
    Request body:
    {
        "user_id": "hashed_user_123",
        "context": {
            "hour": 14,
            "day_of_week": 1,
            "is_weekend": false,
            "battery_level": 75,
            "is_charging": false,
            "network_type": "WiFi"
        },
        "recent_apps": [
            {"app": "Calendar", "timestamp": "2024-11-25T14:25:00Z"}
        ]
    }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract context
        context = data.get('context', {})
        user_id = data.get('user_id', 'anonymous')
        
        # Add defaults if missing
        context.setdefault('hour', datetime.now().hour)
        context.setdefault('day_of_week', datetime.now().weekday())
        context.setdefault('is_weekend', datetime.now().weekday() >= 5)
        context.setdefault('battery_level', 50)
        context.setdefault('is_charging', False)
        context.setdefault('network_type', 'WiFi')
        
        # Make prediction
        if predictor:
            result = predictor.predict(context, top_k=3)
        else:
            # Demo mode - return mock predictions
            result = {
                'predictions': [
                    {'app': 'Instagram', 'confidence': 0.65, 'rank': 1},
                    {'app': 'WhatsApp', 'confidence': 0.20, 'rank': 2},
                    {'app': 'Chrome', 'confidence': 0.10, 'rank': 3}
                ],
                'inference_time_ms': 25,
                'model_version': 'demo'
            }
        
        # Add metadata
        response = {
            'user_id': user_id,
            'predictions': result['predictions'],
            'model_version': result.get('model_version', 'v1.0'),
            'inference_time_ms': result.get('inference_time_ms', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Make batch predictions
    
    Request body:
    {
        "requests": [
            {
                "user_id": "user_1",
                "context": {...}
            },
            {
                "user_id": "user_2",
                "context": {...}
            }
        ]
    }
    """
    try:
        data = request.get_json()
        requests_list = data.get('requests', [])
        
        if not requests_list:
            return jsonify({'error': 'No requests provided'}), 400
        
        results = []
        for req in requests_list:
            context = req.get('context', {})
            user_id = req.get('user_id', 'anonymous')
            
            if predictor:
                result = predictor.predict(context, top_k=3)
            else:
                result = {
                    'predictions': [
                        {'app': 'Instagram', 'confidence': 0.65, 'rank': 1}
                    ],
                    'inference_time_ms': 25,
                    'model_version': 'demo'
                }
            
            results.append({
                'user_id': user_id,
                'predictions': result['predictions']
            })
        
        return jsonify({
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if not predictor:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'model_type': 'XGBoost',
        'version': 'v1.0',
        'num_features': len(predictor.feature_columns),
        'num_apps': len(predictor.app_decoder),
        'features': predictor.feature_columns[:10],  # First 10 features
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run App Prediction API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("iPhone App Prediction API")
    print("="*60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print(f"Model loaded: {predictor is not None}")
    print("="*60 + "\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
