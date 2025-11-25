"""
Complete Pipeline Runner
Runs the entire ML pipeline from data generation to model training
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "="*60)
    print(f"STEP: {description}")
    print("="*60)
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error in: {description}")
        sys.exit(1)
    
    print(f"\n✅ Completed: {description}")


def main():
    parser = argparse.ArgumentParser(description='Run complete ML pipeline')
    parser.add_argument('--num-users', type=int, default=1000, help='Number of users')
    parser.add_argument('--days', type=int, default=30, help='Number of days')
    parser.add_argument('--skip-data', action='store_true', help='Skip data generation')
    parser.add_argument('--skip-features', action='store_true', help='Skip feature engineering')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("iPhone App Prediction - Complete Pipeline")
    print("="*60)
    
    # Step 1: Generate data
    if not args.skip_data:
        cmd = f"python src/data_ingestion/generate_data.py --num-users {args.num_users} --days {args.days}"
        run_command(cmd, "Generate Synthetic Data")
    else:
        print("\n⏭️  Skipping data generation")
    
    # Step 2: Engineer features
    if not args.skip_features:
        cmd = "python src/feature_engineering/feature_engineer.py"
        run_command(cmd, "Engineer Features")
    else:
        print("\n⏭️  Skipping feature engineering")
    
    # Step 3: Train model
    if not args.skip_training:
        cmd = "python src/models/train_xgboost.py"
        run_command(cmd, "Train XGBoost Model")
    else:
        print("\n⏭️  Skipping model training")
    
    # Step 4: Test predictor
    print("\n" + "="*60)
    print("STEP: Test Predictor")
    print("="*60)
    
    try:
        from src.models.predictor import AppPredictor
        
        predictor = AppPredictor('models/xgboost_v1.pkl')
        
        test_context = {
            'hour': 14,
            'day_of_week': 1,
            'battery_level': 75,
            'network_type': 'WiFi'
        }
        
        result = predictor.predict(test_context, top_k=3)
        
        print("\nTest Prediction:")
        for pred in result['predictions']:
            print(f"  {pred['rank']}. {pred['app']:<20} ({pred['confidence']:.1%})")
        
        print(f"\nInference time: {result['inference_time_ms']}ms")
        print("\n✅ Predictor test successful")
        
    except Exception as e:
        print(f"\n⚠️  Warning: Could not test predictor: {e}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run API server: python src/api/app.py")
    print("  2. Test API: curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d @sample_request.json")
    print("  3. View model info: curl http://localhost:5000/model/info")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
