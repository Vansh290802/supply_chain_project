import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn

class MaintenancePredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
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
        mlflow.set_experiment("maintenance_prediction")
        
        try:
            with mlflow.start_run(run_name="maintenance_prediction"):
                # Log model parameters
                mlflow.log_params({
                    "model_type": "GradientBoostingRegressor",
                    "n_estimators": self.model.n_estimators,
                    "learning_rate": self.model.learning_rate,
                    "max_depth": self.model.max_depth,
                    "random_state": self.model.random_state
                })
                
                # Calculate vehicle stress score
                df['Vehicle_Stress_Score'] = (
                    df['Fuel_Consumption_Rate'] * 0.3 +
                    df['IoT_Temperature'] / 100 * 0.2 +
                    (1 - df['Driver_Behavior_Score']) * 0.25 +
                    df['Fatigue_Monitoring_Score'] * 0.25
                )
                
                # Log stress score weights
                mlflow.log_params({
                    "fuel_weight": 0.3,
                    "temp_weight": 0.2,
                    "driver_weight": 0.25,
                    "fatigue_weight": 0.25
                })
                
                # Prepare features and target
                features = self.prepare_maintenance_features(df)
                target = df['Vehicle_Stress_Score']
                
                # Train model and make predictions
                self.model.fit(features, target)
                maintenance_prob = self.model.predict(features)
                
                # Calculate metrics
                mse = mean_squared_error(target, maintenance_prob)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(target, maintenance_prob)
                r2 = r2_score(target, maintenance_prob)
                cv_scores = cross_val_score(self.model, features, target, cv=5)
                
                # Log metrics
                mlflow.log_metrics({
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "cv_score_mean": cv_scores.mean(),
                    "cv_score_std": cv_scores.std()
                })
                
                # Log feature importance
                for idx, feature in enumerate(features.columns):
                    mlflow.log_metric(f"feature_importance_{feature}", 
                                    self.model.feature_importances_[idx])
                
                # Log model
                mlflow.sklearn.log_model(self.model, "maintenance_prediction_model")
                
                prediction_metrics = {
                    'predictions': maintenance_prob,
                    'metrics': {
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'cv_score': cv_scores.mean()
                    },
                    'feature_importance': dict(zip(features.columns, 
                                                 self.model.feature_importances_))
                }
                
                return prediction_metrics
            
        except Exception as e:
            print(f"Error in predict_maintenance_needs: {str(e)}")
            raise
    
    def analyze_vehicle_health(self, df):
        """Analyze vehicle health indicators"""
        mlflow.set_experiment("maintenance_prediction")
        
        try:
            with mlflow.start_run(run_name="vehicle_health_analysis"):
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
                
                # Calculate health metrics
                health_metrics = {
                    'avg_fuel_consumption': df['Fuel_Consumption_Rate'].mean(),
                    'high_temp_incidents': (df['IoT_Temperature'] > df['IoT_Temperature'].quantile(0.95)).sum(),
                    'poor_driving_behavior': (df['Driver_Behavior_Score'] < 0.6).sum(),
                    'high_fatigue_incidents': (df['Fatigue_Monitoring_Score'] > 0.7).sum()
                }
                
                # Calculate additional statistics
                health_metrics.update({
                    'fuel_consumption_std': df['Fuel_Consumption_Rate'].std(),
                    'temp_variation': df['IoT_Temperature'].std(),
                    'driver_score_avg': df['Driver_Behavior_Score'].mean(),
                    'fatigue_score_avg': df['Fatigue_Monitoring_Score'].mean()
                })
                
                # Log metrics
                mlflow.log_metrics(health_metrics)
                
                # Log thresholds used
                mlflow.log_params({
                    "high_temp_threshold": df['IoT_Temperature'].quantile(0.95),
                    "poor_driving_threshold": 0.6,
                    "high_fatigue_threshold": 0.7
                })
                
                return health_metrics
            
        except Exception as e:
            print(f"Error in analyze_vehicle_health: {str(e)}")
            raise
    
    def generate_maintenance_schedule(self, df):
        """Generate preventive maintenance schedule based on location and time"""
        mlflow.set_experiment("maintenance_prediction")
        
        try:
            with mlflow.start_run(run_name="maintenance_scheduling"):
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
                
                # Log schedule generation parameters
                mlflow.log_params({
                    "maintenance_score_weights": {
                        "fuel_consumption": 0.3,
                        "temperature": 0.2,
                        "driver_behavior": 0.25,
                        "fatigue": 0.25
                    }
                })
                
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
                
                # Log key metrics
                mlflow.log_metrics({
                    "avg_maintenance_score": maintenance_priority['Maintenance_Score_mean'].mean(),
                    "max_maintenance_score": maintenance_priority['Maintenance_Score_max'].max(),
                    "total_locations": len(maintenance_priority.index.get_level_values('Location_Group').unique()),
                    "total_time_windows": len(maintenance_priority.index.get_level_values('Time_Window').unique())
                })
                
                return maintenance_priority
            
        except Exception as e:
            print(f"Error in generate_maintenance_schedule: {str(e)}")
            raise