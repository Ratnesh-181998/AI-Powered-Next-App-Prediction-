"""
Feature Engineering for App Prediction
Extracts real-time, contextual, and historical features from user events
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import argparse
from tqdm import tqdm


class FeatureEngineer:
    """Feature engineering for app prediction"""
    
    def __init__(self, config=None):
        self.config = config or {}
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
    
    def extract_temporal_features(self, df):
        """Extract temporal features"""
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] >= 22)).astype(int)
        return df
    
    def extract_sequence_features(self, df, sequence_length=10):
        """Extract app sequence features"""
        print("Extracting sequence features...")
        
        # Sort by user and timestamp
        df = df.sort_values(['user_id_hash', 'timestamp'])
        
        # Create sequence features
        for i in range(1, min(sequence_length, 6)):
            df[f'app_lag_{i}'] = df.groupby('user_id_hash')['app_name'].shift(i)
            df[f'app_lag_{i}_encoded'] = df[f'app_lag_{i}'].apply(self.encode_app)
        
        # Time since last use for each app
        df['time_since_last_use'] = df.groupby(['user_id_hash', 'app_name'])['timestamp'].diff()
        df['time_since_last_use_minutes'] = df['time_since_last_use'].dt.total_seconds() / 60
        df['time_since_last_use_minutes'] = df['time_since_last_use_minutes'].fillna(1440)  # 24 hours
        
        return df
    
    def extract_historical_features(self, df, window_hours=24):
        """Extract historical usage features"""
        print("Extracting historical features...")
        
        df = df.sort_values(['user_id_hash', 'timestamp'])
        
        # App usage count in last 24 hours
        for app in tqdm(df['app_name'].unique()[:20], desc="Processing top apps"):
            df[f'{app}_count_24h'] = 0
            
            for user in df['user_id_hash'].unique():
                user_mask = df['user_id_hash'] == user
                user_df = df[user_mask].copy()
                
                # Count occurrences in rolling window
                counts = []
                for idx, row in user_df.iterrows():
                    window_start = row['timestamp'] - timedelta(hours=window_hours)
                    window_mask = (
                        (user_df['timestamp'] >= window_start) &
                        (user_df['timestamp'] < row['timestamp']) &
                        (user_df['app_name'] == app)
                    )
                    counts.append(window_mask.sum())
                
                df.loc[user_mask, f'{app}_count_24h'] = counts
        
        # Total app opens in last 24 hours
        df['total_opens_24h'] = df.groupby('user_id_hash')['timestamp'].transform(
            lambda x: x.rolling('24H', on=x).count()
        )
        
        return df
    
    def extract_transition_features(self, df):
        """Extract app transition probabilities"""
        print("Extracting transition features...")
        
        # Create transition pairs
        df['prev_app'] = df.groupby('user_id_hash')['app_name'].shift(1)
        df['transition'] = df['prev_app'] + '_to_' + df['app_name']
        
        # Calculate transition probabilities
        transition_counts = df.groupby(['user_id_hash', 'transition']).size()
        total_transitions = df.groupby('user_id_hash').size()
        
        transition_probs = (transition_counts / total_transitions).reset_index()
        transition_probs.columns = ['user_id_hash', 'transition', 'transition_prob']
        
        df = df.merge(transition_probs, on=['user_id_hash', 'transition'], how='left')
        df['transition_prob'] = df['transition_prob'].fillna(0.01)
        
        return df
    
    def create_target(self, df):
        """Create target variable (next app)"""
        df['next_app'] = df.groupby('user_id_hash')['app_name'].shift(-1)
        df['next_app_encoded'] = df['next_app'].apply(self.encode_app)
        return df
    
    def engineer_features(self, df):
        """Main feature engineering pipeline"""
        print("Starting feature engineering...")
        
        # Fit app encoder
        self.fit_app_encoder(df['app_name'].unique())
        
        # Encode current app
        df['app_encoded'] = df['app_name'].apply(self.encode_app)
        
        # Encode network type
        df['network_encoded'] = df['network_type'].map(self.network_encoder).fillna(0)
        
        # Extract features
        df = self.extract_temporal_features(df)
        df = self.extract_sequence_features(df)
        df = self.extract_historical_features(df)
        df = self.extract_transition_features(df)
        
        # Create target
        df = self.create_target(df)
        
        # Drop rows with missing target
        df = df.dropna(subset=['next_app'])
        
        print(f"✅ Feature engineering complete. Shape: {df.shape}")
        
        return df
    
    def get_feature_columns(self):
        """Get list of feature columns"""
        feature_cols = [
            # Temporal features
            'hour', 'day_of_week', 'is_weekend',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'is_morning', 'is_afternoon', 'is_evening', 'is_night',
            
            # Device context
            'battery_level', 'is_charging', 'network_encoded',
            
            # Current app
            'app_encoded', 'session_duration',
            
            # Sequence features
            'app_lag_1_encoded', 'app_lag_2_encoded', 'app_lag_3_encoded',
            'time_since_last_use_minutes',
            
            # Transition features
            'transition_prob',
            
            # Historical features
            'total_opens_24h'
        ]
        
        # Add app-specific counts
        feature_cols.extend([col for col in self.feature_df.columns if col.endswith('_count_24h')])
        
        return feature_cols


def main():
    parser = argparse.ArgumentParser(description='Engineer features from user events')
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
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df)
    
    # Save features
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path, index=False)
    
    print(f"\n✅ Saved features to: {output_path}")
    print(f"   Shape: {df_features.shape}")
    print(f"   Features: {len(engineer.get_feature_columns())}")


if __name__ == '__main__':
    main()
