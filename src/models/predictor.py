"""
App Predictor - Inference Module
Loads trained model and makes predictions
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import time


class AppPredictor:
    """Make app predictions using trained model"""
    
    def __init__(self, model_path='models/xgboost_v1.pkl'):
        """Initialize predictor with trained model"""
        print(f"Loading model from {model_path}...")
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.app_decoder = model_data['app_decoder']
        self.config = model_data.get('config', {})
        
        print(f"[OK] Model loaded successfully")
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Apps: {len(self.app_decoder)}")
    
    def prepare_features(self, context: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features from context"""
        # Create feature dictionary with defaults
        features = {}
        
        # Temporal features
        hour = context.get('hour', 12)
        features['hour'] = hour
        features['day_of_week'] = context.get('day_of_week', 1)
        features['is_weekend'] = int(context.get('is_weekend', False))
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # Time of day
        features['is_morning'] = int(6 <= hour < 12)
        features['is_afternoon'] = int(12 <= hour < 18)
        features['is_evening'] = int(18 <= hour < 22)
        features['is_night'] = int(hour < 6 or hour >= 22)
        
        # Device context
        features['battery_level'] = context.get('battery_level', 50)
        features['is_charging'] = int(context.get('is_charging', False))
        
        # Network type encoding
        network_map = {'WiFi': 1, '4G': 2, '3G': 3, 'Offline': 0}
        features['network_encoded'] = network_map.get(context.get('network_type', 'WiFi'), 1)
        
        # App encoding (simplified - in production would use actual encoder)
        features['app_encoded'] = context.get('app_encoded', 0)
        features['session_duration'] = context.get('session_duration', 5)
        
        # Sequence features
        features['app_lag_1_encoded'] = context.get('app_lag_1_encoded', 0)
        features['app_lag_2_encoded'] = context.get('app_lag_2_encoded', 0)
        features['app_lag_3_encoded'] = context.get('app_lag_3_encoded', 0)
        features['time_since_last_use_minutes'] = context.get('time_since_last_use_minutes', 60)
        
        # Transition probability
        features['transition_prob'] = context.get('transition_prob', 0.1)
        
        # Historical features
        features['total_opens_24h'] = context.get('total_opens_24h', 50)
        
        # App-specific counts (fill with defaults)
        for col in self.feature_columns:
            if col.endswith('_count_24h') and col not in features:
                features[col] = context.get(col, 0)
        
        # Create DataFrame with only the features used in training
        feature_df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        # Select only the features used in training, in the correct order
        feature_df = feature_df[self.feature_columns]
        
        return feature_df
    
    def predict(self, context: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """Make prediction and return top-k apps"""
        start_time = time.time()
        
        # Prepare features
        features = self.prepare_features(context)
        
        # Get predictions
        probabilities = self.model.predict_proba(features)[0]
        
        # Get top-k predictions
        top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        predictions = []
        for rank, idx in enumerate(top_k_indices, 1):
            app_name = self.app_decoder.get(idx, f'Unknown_App_{idx}')
            confidence = float(probabilities[idx])
            
            predictions.append({
                'app': app_name,
                'confidence': round(confidence, 4),
                'rank': rank
            })
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return {
            'predictions': predictions,
            'inference_time_ms': round(inference_time, 2),
            'model_version': 'xgboost_v1'
        }
    
    def predict_batch(self, contexts: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Make batch predictions"""
        results = []
        for context in contexts:
            result = self.predict(context, top_k=top_k)
            results.append(result)
        return results


def demo():
    """Demo prediction"""
    # Initialize predictor
    predictor = AppPredictor('models/xgboost_v1.pkl')
    
    # Example contexts
    contexts = [
        {
            'hour': 9,
            'day_of_week': 1,
            'is_weekend': False,
            'battery_level': 85,
            'is_charging': False,
            'network_type': 'WiFi',
            'session_duration': 5,
            'total_opens_24h': 50
        },
        {
            'hour': 14,
            'day_of_week': 1,
            'is_weekend': False,
            'battery_level': 60,
            'is_charging': False,
            'network_type': '4G',
            'session_duration': 3,
            'total_opens_24h': 45
        },
        {
            'hour': 20,
            'day_of_week': 6,
            'is_weekend': True,
            'battery_level': 40,
            'is_charging': True,
            'network_type': 'WiFi',
            'session_duration': 10,
            'total_opens_24h': 60
        }
    ]
    
    # Make predictions
    print("\n" + "="*60)
    print("DEMO PREDICTIONS")
    print("="*60)
    
    for i, context in enumerate(contexts, 1):
        print(f"\nScenario {i}:")
        print(f"  Time: {context['hour']}:00, {'Weekend' if context['is_weekend'] else 'Weekday'}")
        print(f"  Battery: {context['battery_level']}%, Network: {context['network_type']}")
        
        result = predictor.predict(context, top_k=3)
        
        print(f"\n  Predictions (in {result['inference_time_ms']}ms):")
        for pred in result['predictions']:
            print(f"    {pred['rank']}. {pred['app']:<20} ({pred['confidence']:.1%})")


if __name__ == '__main__':
    demo()
