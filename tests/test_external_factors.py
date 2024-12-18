import pandas as pd
import numpy as np
from src.external_factors.external_factors import ExternalFactorsAnalyzer
import mlflow
from datetime import datetime, timedelta

def generate_sample_data(n_samples=1000):
    """Generate sample data for testing"""
    np.random.seed(42)
    
    # Generate timestamps
    base_timestamp = datetime(2023, 1, 1)
    timestamps = [base_timestamp + timedelta(hours=x) for x in range(n_samples)]
    
    # Generate sample data
    data = {
        'Timestamp': timestamps,
        'Weather_Condition_Severity': np.random.uniform(0, 1, n_samples),
        'Traffic_Congestion_Level': np.random.randint(0, 11, n_samples),
        'Port_Congestion_Level': np.random.randint(0, 11, n_samples),
        'Fuel_Consumption_Rate': np.random.normal(8, 2, n_samples),
        'Loading_Unloading_Time': np.random.normal(2, 0.5, n_samples),
        'Delay_Probability': np.random.uniform(0, 1, n_samples),
        'ETA_Variation': np.random.normal(1, 0.5, n_samples),
        'Delivery_Time_Deviation': np.random.normal(0, 2, n_samples),
        'Customs_Clearance_Time': np.random.normal(5, 1, n_samples),
        'Lead_Time': np.random.normal(10, 2, n_samples),
        'Order_Fulfillment_Status': np.random.randint(0, 2, n_samples)
    }
    
    return pd.DataFrame(data)

def test_external_factors_analysis():
    """Test the ExternalFactorsAnalyzer with MLflow tracking"""
    print("Starting External Factors Analysis Test...")
    
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data()
    print(f"Generated {len(df)} samples")
    
    # Initialize analyzer
    analyzer = ExternalFactorsAnalyzer()
    
    # Test temporal pattern analysis
    print("\nTesting temporal pattern analysis...")
    temporal_patterns = analyzer.analyze_temporal_patterns(df)
    print("Temporal patterns keys:", temporal_patterns.keys())
    
    # Test weather impact analysis
    print("\nTesting weather impact analysis...")
    weather_analysis = analyzer.analyze_weather_impact(df)
    print("Weather analysis keys:", weather_analysis.keys())
    
    # Test traffic pattern analysis
    print("\nTesting traffic pattern analysis...")
    traffic_analysis = analyzer.analyze_traffic_patterns(df)
    print("Traffic analysis keys:", traffic_analysis.keys())
    
    # Test port operations analysis
    print("\nTesting port operations analysis...")
    port_analysis = analyzer.analyze_port_operations(df)
    print("Port analysis keys:", port_analysis.keys())
    
    # Test combined impact analysis
    print("\nTesting combined impact analysis...")
    combined_analysis = analyzer.analyze_combined_impact(df)
    print("Combined analysis keys:", combined_analysis.keys())
    
    # Test prediction with MLflow tracking
    print("\nTesting prediction with MLflow tracking...")
    try:
        prediction_metrics = analyzer.predict_delivery_impact(df)
        print("\nPrediction Results:")
        print("Delay Predictor Metrics:")
        print(f"- R² Score: {prediction_metrics['delay_metrics']['r2']:.4f}")
        print(f"- RMSE: {prediction_metrics['delay_metrics']['rmse']:.4f}")
        print(f"- MAE: {prediction_metrics['delay_metrics']['mae']:.4f}")
        print(f"- CV Score: {prediction_metrics['delay_metrics']['cv_score']:.4f}")
        
        print("\nTime Deviation Predictor Metrics:")
        print(f"- R² Score: {prediction_metrics['deviation_metrics']['r2']:.4f}")
        print(f"- RMSE: {prediction_metrics['deviation_metrics']['rmse']:.4f}")
        print(f"- MAE: {prediction_metrics['deviation_metrics']['mae']:.4f}")
        print(f"- CV Score: {prediction_metrics['deviation_metrics']['cv_score']:.4f}")
        
        # Print feature importance
        print("\nFeature Importance:")
        print(prediction_metrics['feature_importance'])
        
        print("\nMLflow tracking successful!")
        
    except Exception as e:
        print(f"Error during prediction and MLflow tracking: {str(e)}")
    
    # Test recommendations
    print("\nTesting recommendations generation...")
    recommendations = analyzer.generate_recommendations(df)
    print(f"Generated {len(recommendations)} recommendations")
    for rec in recommendations:
        print(f"\nFactor: {rec['factor']}")
        print(f"Issue: {rec['issue']}")
        print(f"Recommendation: {rec['recommendation']}")

if __name__ == "__main__":
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    test_external_factors_analysis()