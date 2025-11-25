"""
XGBoost Model Training for App Prediction
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path
import argparse
import yaml
import json
from datetime import datetime


class XGBoostTrainer:
    """Train XGBoost model for app prediction"""
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.feature_columns = None
        self.app_decoder = None
        
    def prepare_data(self, df):
        """Prepare data for training"""
        print("Preparing data...")
        
        # Get feature columns (exclude target and metadata)
        exclude_cols = [
            'event_id', 'user_id_hash', 'timestamp', 'app_name', 
            'next_app', 'location_bucket', 'transition', 'prev_app',
            'time_since_last_use'
        ]
        
        # Get all columns that are not in exclude list
        self.feature_columns = [
            col for col in df.columns 
            if col not in exclude_cols and not col.startswith('app_lag_') or col.endswith('_encoded')
        ]
        
        # Ensure we have numeric features
        self.feature_columns = [
            col for col in self.feature_columns 
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']
        ]
        
        print(f"Using {len(self.feature_columns)} features")
        
        X = df[self.feature_columns]
        y = df['next_app_encoded']
        
        # Create app decoder
        self.app_decoder = {
            int(code): app 
            for app, code in zip(df['next_app'], df['next_app_encoded'])
        }
        
        return X, y
    
    def train_test_split_time(self, X, y, df, test_size=0.2):
        """Time-based train-test split"""
        # Sort by timestamp
        timestamps = pd.to_datetime(df['timestamp'])
        sorted_idx = timestamps.argsort()
        
        split_idx = int(len(X) * (1 - test_size))
        
        train_idx = sorted_idx[:split_idx]
        test_idx = sorted_idx[split_idx:]
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        # Get hyperparameters from config
        params = self.config['xgboost'].copy()
        
        # Create model
        self.model = xgb.XGBClassifier(**params)
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        print("[OK] Training complete")
        
        return self.model
    
    def evaluate(self, X_test, y_test, top_k=[1, 3, 5]):
        """Evaluate model performance"""
        print("\nEvaluating model...")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Top-1 accuracy
        top1_acc = accuracy_score(y_test, y_pred)
        print(f"Top-1 Accuracy: {top1_acc:.4f}")
        
        # Top-K accuracy
        for k in top_k:
            if k > 1:
                top_k_pred = np.argsort(y_pred_proba, axis=1)[:, -k:]
                top_k_acc = np.mean([y_test.iloc[i] in top_k_pred[i] for i in range(len(y_test))])
                print(f"Top-{k} Accuracy: {top_k_acc:.4f}")
        
        # Classification report for top 10 apps
        top_apps = y_test.value_counts().head(10).index.tolist()
        mask = y_test.isin(top_apps)
        
        if mask.sum() > 0:
            print("\nClassification Report (Top 10 Apps):")
            print(classification_report(
                y_test[mask], 
                y_pred[mask],
                labels=top_apps,
                target_names=[self.app_decoder.get(int(i), f'App_{i}') for i in top_apps],
                zero_division=0
            ))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return {
            'top1_accuracy': float(top1_acc),
            'top3_accuracy': float(top_k_acc) if len(top_k) > 1 else None,
            'feature_importance': feature_importance.to_dict('records')
        }
    
    def save_model(self, output_path='models/xgboost_v1.pkl'):
        """Save trained model"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'app_decoder': self.app_decoder,
            'config': self.config,
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, output_path)
        print(f"\n[OK] Model saved to: {output_path}")
        
        # Save metadata separately
        metadata_path = output_path.parent / 'model_metadata.json'
        metadata = {
            'model_type': 'XGBoost',
            'num_features': len(self.feature_columns),
            'num_classes': len(self.app_decoder),
            'trained_at': datetime.now().isoformat(),
            'config': self.config['xgboost']
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[OK] Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model')
    parser.add_argument('--data', type=str, default='data/processed/features.parquet',
                        help='Input features file')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Config file')
    parser.add_argument('--output', type=str, default='models/xgboost_v1.pkl',
                        help='Output model file')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df):,} samples")
    
    # Initialize trainer
    trainer = XGBoostTrainer(config_path=args.config)
    
    # Prepare data
    X, y = trainer.prepare_data(df)
    
    # Train-test split (time-based)
    X_train, X_test, y_train, y_test = trainer.train_test_split_time(X, y, df, test_size=0.2)
    
    # Further split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train):,}")
    print(f"  Validation: {len(X_val):,}")
    print(f"  Test: {len(X_test):,}")
    
    # Train model
    trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save model
    trainer.save_model(output_path=args.output)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)


if __name__ == '__main__':
    main()
