import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

class MaintenancePredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor(random_state=42)
        
    def prepare_maintenance_features(self, df):
        """Prepare features for maintenance prediction"""
        required_features = [
            'Fuel_Consumption_Rate',
            'IoT_Temperature',
            'Driver_Behavior_Score',
            'Fatigue_Monitoring_Score',
            'Route_Risk_Level'
        ]
        
        # Verify all required features are present
        missing_features = [col for col in required_features if col not in df.columns]
        if missing_features:
            raise KeyError(f"Missing required features: {missing_features}")
            
        return df[required_features]
    
    def predict_maintenance_needs(self, df):
        """Predict maintenance requirements based on vehicle conditions"""
        try:
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
            
        except KeyError as e:
            print(f"Error in predict_maintenance_needs: {str(e)}")
            raise
    
    def analyze_vehicle_health(self, df):
        """Analyze vehicle health indicators"""
        try:
            required_columns = [
                'Fuel_Consumption_Rate',
                'IoT_Temperature',
                'Driver_Behavior_Score',
                'Fatigue_Monitoring_Score'
            ]
            
            # Verify all required columns are present
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise KeyError(f"Missing required columns: {missing_columns}")
            
            health_metrics = {
                'avg_fuel_consumption': df['Fuel_Consumption_Rate'].mean(),
                'high_temp_incidents': (df['IoT_Temperature'] > df['IoT_Temperature'].quantile(0.95)).sum(),
                'poor_driving_behavior': (df['Driver_Behavior_Score'] < 0.6).sum(),
                'high_fatigue_incidents': (df['Fatigue_Monitoring_Score'] > 0.7).sum()
            }
            
            return health_metrics
            
        except Exception as e:
            print(f"Error in analyze_vehicle_health: {str(e)}")
            raise
    
    def generate_maintenance_schedule(self, df):
        """Generate preventive maintenance schedule based on location and time"""
        try:
            required_columns = [
                'Timestamp',
                'Vehicle_GPS_Latitude',
                'Vehicle_GPS_Longitude',
                'Fuel_Consumption_Rate',
                'IoT_Temperature',
                'Driver_Behavior_Score',
                'Fatigue_Monitoring_Score',
                'Route_Risk_Level'
            ]
            
            # Verify all required columns are present
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise KeyError(f"Missing required columns: {missing_columns}")
            
            # Create location-based grouping
            df['Location_Group'] = df.apply(
                lambda row: f"loc_{int(row['Vehicle_GPS_Latitude'] * 100)}_{int(row['Vehicle_GPS_Longitude'] * 100)}", 
                axis=1
            )
            
            # Calculate maintenance score
            df['Maintenance_Score'] = (
                df['Fuel_Consumption_Rate'] / df['Fuel_Consumption_Rate'].max() * 0.3 +
                (df['IoT_Temperature'] - df['IoT_Temperature'].min()) / 
                (df['IoT_Temperature'].max() - df['IoT_Temperature'].min()) * 0.2 +
                (1 - df['Driver_Behavior_Score']) * 0.25 +
                df['Fatigue_Monitoring_Score'] * 0.25
            )
            
            # Extract time window
            df['Time_Window'] = pd.to_datetime(df['Timestamp']).dt.strftime('%Y-%m-%d')
            
            # Group by location and time window
            maintenance_priority = df.groupby(['Location_Group', 'Time_Window']).agg({
                'Maintenance_Score': ['mean', 'max'],
                'Fuel_Consumption_Rate': 'mean',
                'IoT_Temperature': 'mean',
                'Driver_Behavior_Score': 'mean',
                'Fatigue_Monitoring_Score': 'mean',
                'Route_Risk_Level': 'mean'
            })
            
            # Flatten column names
            maintenance_priority.columns = [
                f'{col[0]}_{col[1]}' for col in maintenance_priority.columns
            ]
            
            # Sort by maintenance score
            maintenance_priority = maintenance_priority.sort_values(
                'Maintenance_Score_mean', 
                ascending=False
            )
            
            return maintenance_priority
            
        except Exception as e:
            print(f"Error in generate_maintenance_schedule: {str(e)}")
            raise