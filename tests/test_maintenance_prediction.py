import pandas as pd
import numpy as np
from src.maintenance_prediction.maintenance_prediction import MaintenancePredictor
import mlflow
from datetime import datetime, timedelta

def generate_sample_data(n_samples=1000):
    """Generate sample data for testing maintenance prediction"""
    np.random.seed(42)
    
    # Generate timestamps
    base_timestamp = datetime(2023, 1, 1)
    timestamps = [base_timestamp + timedelta(hours=x) for x in range(n_samples)]
    
    # Generate GPS coordinates (Southern California area)
    lat_range = (32.5, 34.5)
    lon_range = (-118.5, -116.5)
    
    data = {
        'Timestamp': timestamps,
        'Vehicle_GPS_Latitude': np.random.uniform(lat_range[0], lat_range[1], n_samples),
        'Vehicle_GPS_Longitude': np.random.uniform(lon_range[0], lon_range[1], n_samples),
        'Fuel_Consumption_Rate': np.random.normal(8, 2, n_samples),
        'IoT_Temperature': np.random.normal(60, 15, n_samples),
        'Driver_Behavior_Score': np.random.uniform(0.3, 1.0, n_samples),
        'Fatigue_Monitoring_Score': np.random.uniform(0, 1, n_samples),
        'Route_Risk_Level': np.random.randint(0, 11, n_samples)
    }
    
    # Ensure values are within reasonable ranges
    data['Fuel_Consumption_Rate'] = np.clip(data['Fuel_Consumption_Rate'], 0, None)
    data['IoT_Temperature'] = np.clip(data['IoT_Temperature'], 0, 100)
    data['Driver_Behavior_Score'] = np.clip(data['Driver_Behavior_Score'], 0, 1)
    data['Fatigue_Monitoring_Score'] = np.clip(data['Fatigue_Monitoring_Score'], 0, 1)
    
    return pd.DataFrame(data)

def print_separator():
    print("\n" + "="*80 + "\n")

def test_maintenance_prediction():
    """Test the MaintenancePredictor with MLflow tracking"""
    print("Starting Maintenance Prediction Test...")
    print_separator()
    
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data()
    print(f"Generated {len(df)} samples")
    print_separator()
    
    # Initialize maintenance predictor
    predictor = MaintenancePredictor()
    
    # Test maintenance needs prediction
    print("Testing maintenance needs prediction...")
    try:
        prediction_results = predictor.predict_maintenance_needs(df)
        
        print("\nPrediction Metrics:")
        for metric, value in prediction_results['metrics'].items():
            print(f"- {metric}: {value:.4f}")
        
        print("\nFeature Importance:")
        for feature, importance in prediction_results['feature_importance'].items():
            print(f"- {feature}: {importance:.4f}")
            
        print(f"\nPredicted maintenance probabilities (first 5):")
        print(prediction_results['predictions'][:5])
    except Exception as e:
        print(f"Error during maintenance prediction: {str(e)}")
    print_separator()
    
    # Test vehicle health analysis
    print("Testing vehicle health analysis...")
    try:
        health_metrics = predictor.analyze_vehicle_health(df)
        
        print("\nVehicle Health Metrics:")
        for metric, value in health_metrics.items():
            if isinstance(value, (int, np.integer)):
                print(f"- {metric}: {value:,}")
            else:
                print(f"- {metric}: {value:.4f}")
    except Exception as e:
        print(f"Error during vehicle health analysis: {str(e)}")
    print_separator()
    
    # Test maintenance schedule generation
    print("Testing maintenance schedule generation...")
    try:
        maintenance_schedule = predictor.generate_maintenance_schedule(df)
        
        print("\nMaintenance Priority Schedule (top 5 locations/times):")
        print(maintenance_schedule.head())
        
        print("\nSchedule Statistics:")
        print(f"- Total locations: {len(maintenance_schedule.index.get_level_values('Location_Group').unique())}")
        print(f"- Total time windows: {len(maintenance_schedule.index.get_level_values('Time_Window').unique())}")
        print(f"- Average maintenance score: {maintenance_schedule['Maintenance_Score_mean'].mean():.4f}")
        print(f"- Maximum maintenance score: {maintenance_schedule['Maintenance_Score_max'].max():.4f}")
        
        # Print example high-priority maintenance location
        high_priority = maintenance_schedule.iloc[0]
        print("\nHighest Priority Maintenance Location/Time:")
        print(f"Location/Time: {high_priority.name}")
        print(f"- Maintenance Score (mean): {high_priority['Maintenance_Score_mean']:.4f}")
        print(f"- Maintenance Score (max): {high_priority['Maintenance_Score_max']:.4f}")
        print(f"- Average Fuel Consumption: {high_priority['Fuel_Consumption_Rate_mean']:.4f}")
        print(f"- Average Temperature: {high_priority['IoT_Temperature_mean']:.4f}")
        print(f"- Average Driver Score: {high_priority['Driver_Behavior_Score_mean']:.4f}")
        print(f"- Average Fatigue Score: {high_priority['Fatigue_Monitoring_Score_mean']:.4f}")
        
    except Exception as e:
        print(f"Error during maintenance schedule generation: {str(e)}")
    print_separator()

if __name__ == "__main__":
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print_separator()
    test_maintenance_prediction()