"""
Data Generator for iPhone App Prediction System
Generates synthetic user behavior data for training and testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path
import argparse
from tqdm import tqdm


class UserBehaviorGenerator:
    """Generate synthetic user behavior data"""
    
    def __init__(self, num_users=1000, num_days=30, seed=42):
        self.num_users = num_users
        self.num_days = num_days
        self.seed = seed
        np.random.seed(seed)
        
        # Top 50 apps
        self.apps = [
            'Instagram', 'WhatsApp', 'Gmail', 'Chrome', 'YouTube',
            'Calendar', 'Maps', 'Spotify', 'Twitter', 'Facebook',
            'Slack', 'Zoom', 'Netflix', 'Amazon', 'Uber',
            'Photos', 'Camera', 'Settings', 'Messages', 'Phone',
            'Safari', 'Music', 'Podcast', 'News', 'Weather',
            'Notes', 'Reminders', 'Clock', 'Calculator', 'Wallet',
            'Health', 'Fitness', 'LinkedIn', 'TikTok', 'Snapchat',
            'Reddit', 'Pinterest', 'Telegram', 'Discord', 'Twitch',
            'Medium', 'Kindle', 'Audible', 'Duolingo', 'Headspace',
            'Banking', 'Trading', 'Food Delivery', 'Shopping', 'Games'
        ]
        
        # User personas
        self.personas = {
            'professional': {
                'morning': ['Gmail', 'Calendar', 'News', 'Weather', 'Slack'],
                'afternoon': ['Slack', 'Zoom', 'Gmail', 'LinkedIn', 'Chrome'],
                'evening': ['Instagram', 'YouTube', 'Netflix', 'Twitter', 'Reddit'],
                'night': ['Instagram', 'YouTube', 'Reddit', 'Twitter', 'Netflix']
            },
            'student': {
                'morning': ['Instagram', 'WhatsApp', 'YouTube', 'Twitter', 'TikTok'],
                'afternoon': ['Chrome', 'Notes', 'Calendar', 'Zoom', 'Gmail'],
                'evening': ['Instagram', 'TikTok', 'YouTube', 'Snapchat', 'Netflix'],
                'night': ['Instagram', 'TikTok', 'YouTube', 'Twitter', 'Reddit']
            },
            'casual': {
                'morning': ['WhatsApp', 'Instagram', 'News', 'Weather', 'Facebook'],
                'afternoon': ['WhatsApp', 'Instagram', 'Facebook', 'YouTube', 'Chrome'],
                'evening': ['Netflix', 'YouTube', 'Instagram', 'Facebook', 'WhatsApp'],
                'night': ['Netflix', 'YouTube', 'Instagram', 'Reddit', 'Twitter']
            }
        }
        
    def hash_user_id(self, user_id):
        """Hash user ID for privacy"""
        return hashlib.sha256(str(user_id).encode()).hexdigest()[:16]
    
    def get_time_period(self, hour):
        """Get time period from hour"""
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    def generate_user_profile(self, user_id):
        """Generate user profile"""
        persona = np.random.choice(list(self.personas.keys()))
        return {
            'user_id': user_id,
            'user_id_hash': self.hash_user_id(user_id),
            'persona': persona,
            'favorite_apps': np.random.choice(self.apps, size=10, replace=False).tolist()
        }
    
    def generate_app_open_event(self, user_profile, timestamp, context):
        """Generate single app open event"""
        hour = timestamp.hour
        time_period = self.get_time_period(hour)
        persona = user_profile['persona']
        
        # Get likely apps for this time period
        likely_apps = self.personas[persona][time_period]
        
        # 70% chance to use persona-specific app, 30% random
        if np.random.random() < 0.7:
            app = np.random.choice(likely_apps)
        else:
            app = np.random.choice(self.apps)
        
        return {
            'event_id': f"evt_{np.random.randint(100000, 999999)}",
            'user_id_hash': user_profile['user_id_hash'],
            'timestamp': timestamp.isoformat(),
            'app_name': app,
            'hour': hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': timestamp.weekday() >= 5,
            'location_bucket': context['location_bucket'],
            'battery_level': context['battery_level'],
            'is_charging': context['is_charging'],
            'network_type': context['network_type'],
            'session_duration': np.random.randint(1, 30)  # minutes
        }
    
    def generate_context(self, hour):
        """Generate device context"""
        # Battery level decreases during day, increases when charging
        if hour < 8:
            battery_level = np.random.randint(80, 100)
            is_charging = True
        elif hour < 12:
            battery_level = np.random.randint(60, 90)
            is_charging = False
        elif hour < 18:
            battery_level = np.random.randint(40, 70)
            is_charging = False
        else:
            battery_level = np.random.randint(20, 60)
            is_charging = np.random.choice([True, False], p=[0.3, 0.7])
        
        # Network type based on time and location
        if 9 <= hour < 18:  # Work hours
            network_type = np.random.choice(['WiFi', '4G'], p=[0.7, 0.3])
            location_bucket = f"work_{np.random.randint(1, 10)}"
        elif 18 <= hour < 22:  # Evening
            network_type = np.random.choice(['WiFi', '4G'], p=[0.8, 0.2])
            location_bucket = f"home_{np.random.randint(1, 10)}"
        else:  # Night/Morning
            network_type = 'WiFi'
            location_bucket = f"home_{np.random.randint(1, 10)}"
        
        return {
            'battery_level': battery_level,
            'is_charging': is_charging,
            'network_type': network_type,
            'location_bucket': location_bucket
        }
    
    def generate_user_data(self, user_profile, start_date):
        """Generate data for one user"""
        events = []
        
        for day in range(self.num_days):
            current_date = start_date + timedelta(days=day)
            
            # Generate 20-100 app opens per day
            num_events = np.random.randint(20, 100)
            
            # Generate events throughout the day
            for _ in range(num_events):
                hour = np.random.choice(range(24), p=self._get_hour_distribution())
                minute = np.random.randint(0, 60)
                timestamp = current_date.replace(hour=hour, minute=minute)
                
                context = self.generate_context(hour)
                event = self.generate_app_open_event(user_profile, timestamp, context)
                events.append(event)
        
        return events
    
    def _get_hour_distribution(self):
        """Get probability distribution for hours of day"""
        # People are more active during waking hours
        probs = np.array([
            0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5 (night)
            0.03, 0.05, 0.06, 0.07, 0.08, 0.08,  # 6-11 (morning)
            0.07, 0.06, 0.06, 0.06, 0.06, 0.06,  # 12-17 (afternoon)
            0.07, 0.08, 0.08, 0.07, 0.04, 0.02   # 18-23 (evening/night)
        ])
        return probs / probs.sum()
    
    def generate_dataset(self, output_path='data/raw/user_events.parquet'):
        """Generate complete dataset"""
        print(f"Generating data for {self.num_users} users over {self.num_days} days...")
        
        all_events = []
        start_date = datetime.now() - timedelta(days=self.num_days)
        
        for user_id in tqdm(range(self.num_users), desc="Generating user data"):
            user_profile = self.generate_user_profile(user_id)
            user_events = self.generate_user_data(user_profile, start_date)
            all_events.extend(user_events)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_events)
        
        # Sort by timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Save to parquet
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        
        print(f"\n[OK] Generated {len(df):,} events")
        print(f"[OK] Saved to: {output_path}")
        print(f"\nDataset Info:")
        print(f"  - Users: {df['user_id_hash'].nunique()}")
        print(f"  - Apps: {df['app_name'].nunique()}")
        print(f"  - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  - Avg events per user: {len(df) / df['user_id_hash'].nunique():.0f}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic user behavior data')
    parser.add_argument('--num-users', type=int, default=1000, help='Number of users')
    parser.add_argument('--days', type=int, default=30, help='Number of days')
    parser.add_argument('--output', type=str, default='data/raw/user_events.parquet', 
                        help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    generator = UserBehaviorGenerator(
        num_users=args.num_users,
        num_days=args.days,
        seed=args.seed
    )
    
    generator.generate_dataset(output_path=args.output)


if __name__ == '__main__':
    main()
