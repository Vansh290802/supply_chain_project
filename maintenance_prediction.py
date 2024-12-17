import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

class MaintenancePredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor(random_state=42)
        
    def prepare_maintenance_features(self, df):
        """Prepare features for maintenance prediction"""
        maintenance_features = [
            'Fuel_Consumption_Rate',
            'IoT_Temperature',
            'Driver_Behavior_Score',
            'Fatigue_Monitoring_Score',
            'Route_Risk_Level'
        ]
        
        return df[maintenance_features]
    
    def predict_maintenance_needs(self, df):
        """Predict maintenance requirements based on vehicle conditions"""
        # Calculate vehicle stress score
        df['Vehicle_Stress_Score'] = (
            df['Fuel_Consumption_Rate'] * 0.3 +
            df['IoT_Temperature'] / 100 * 0.2 +
            (1 - df['Driver_Behavior_Score']) * 0.25 +
            df['Fatigue_Monitoring_Score'] * 0.25
        )
        
        # Predict maintenance probability
        features = self.prepare_maintenance_features(df)
        maintenance_prob = self.model.predict(features)
        
        return maintenance_prob
    
    def analyze_vehicle_health(self, df):
        """Analyze vehicle health indicators"""
        health_metrics = {
            'avg_fuel_consumption': df['Fuel_Consumption_Rate'].mean(),
            'high_temp_incidents': (df['IoT_Temperature'] > df['IoT_Temperature'].quantile(0.95)).sum(),
            'poor_driving_behavior': (df['Driver_Behavior_Score'] < 0.6).sum(),
            'high_fatigue_incidents': (df['Fatigue_Monitoring_Score'] > 0.7).sum()
        }
        
        return health_metrics
    
    def generate_maintenance_schedule(self, df):
        """Generate preventive maintenance schedule"""
        # Calculate days since last maintenance (simplified)
        df['Days_Since_Maintenance'] = (
            pd.to_datetime(df['Timestamp']).dt.to_period('D').astype(int) % 30
        )
        
        # Prioritize vehicles for maintenance
        maintenance_priority = df.groupby('Vehicle_ID').agg({
            'Vehicle_Stress_Score': 'mean',
            'Fuel_Consumption_Rate': 'mean',
            'IoT_Temperature': 'mean',
            'Days_Since_Maintenance': 'max'
        }).sort_values('Vehicle_Stress_Score', ascending=False)
        
        return maintenance_priority