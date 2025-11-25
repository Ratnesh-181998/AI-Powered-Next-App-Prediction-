"""
Simplified Feature Engineering for Demo
Fast feature extraction for app prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


class SimplifiedFeatureEngineer:
    """Simplified feature engineering for faster processing"""
    
    def __init__(self):
        self.app_encoder = {}
        self.network_encoder = {'WiFi': 1, '4G': 2, '3G': 3, 'Offline': 0}
        
    def fit_app_encoder(self, apps):
        """Create app encoding"""
        unique_apps = sorted(set(apps))
        self.app_encoder = {app: idx for idx, app in enumerate(unique_apps)}
        return self.app_encoder
    
    def encode_app(self, app):
        """Encode app name to integer"""
        return self.app_encoder.get(app, 0)
    
    def engineer_features(self, df):
        """Main feature engineering pipeline - simplified"""
        print("Starting simplified feature engineering...")
        
        # Fit app encoder
        self.fit_app_encoder(df['app_name'].unique())
        
        # Encode current app
        df['app_encoded'] = df['app_name'].apply(self.encode_app)
        
        # Encode network type
        df['network_encoded'] = df['network_type'].map(self.network_encoder).fillna(0)
        
        # Temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] >= 22)).astype(int)
        
        # Sort by user and timestamp
        df = df.sort_values(['user_id_hash', 'timestamp'])
        
        # Simple sequence features
        df['app_lag_1'] = df.groupby('user_id_hash')['app_name'].shift(1)
        df['app_lag_1_encoded'] = df['app_lag_1'].apply(self.encode_app)
        df['app_lag_2'] = df.groupby('user_id_hash')['app_name'].shift(2)
        df['app_lag_2_encoded'] = df['app_lag_2'].apply(self.encode_app)
        
        # Create target
        df['next_app'] = df.groupby('user_id_hash')['app_name'].shift(-1)
        df['next_app_encoded'] = df['next_app'].apply(self.encode_app)
        
        # Drop rows with missing target
        df = df.dropna(subset=['next_app'])
        
        print(f"[OK] Feature engineering complete. Shape: {df.shape}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Engineer features (simplified)')
    parser.add_argument('--input', type=str, default='data/raw/user_events.parquet',
                        help='Input events file')
    parser.add_argument('--output', type=str, default='data/processed/features.parquet',
                        help='Output features file')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} events")
    
    # Engineer features
    engineer = SimplifiedFeatureEngineer()
    df_features = engineer.engineer_features(df)
    
    # Save features
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path, index=False)
    
    print(f"\n[OK] Saved features to: {output_path}")
    print(f"   Shape: {df_features.shape}")


if __name__ == '__main__':
    main()
