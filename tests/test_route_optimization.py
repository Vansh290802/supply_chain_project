import pandas as pd
import numpy as np
from src.route_optimization.route_optimization import RouteOptimizer
import mlflow
from datetime import datetime, timedelta

def generate_sample_data(n_samples=1000):
    """Generate sample data for testing route optimization"""
    np.random.seed(42)
    
    # Generate timestamps
    base_timestamp = datetime(2023, 1, 1)
    timestamps = [base_timestamp + timedelta(hours=x) for x in range(n_samples)]
    
    # Generate GPS coordinates (Southern California area)
    lat_range = (32.5, 34.5)  # Southern California latitude range
    lon_range = (-118.5, -116.5)  # Southern California longitude range
    
    data = {
        'Timestamp': timestamps,
        'Vehicle_GPS_Latitude': np.random.uniform(lat_range[0], lat_range[1], n_samples),
        'Vehicle_GPS_Longitude': np.random.uniform(lon_range[0], lon_range[1], n_samples),
        'Traffic_Congestion_Level': np.random.randint(0, 11, n_samples),
        'Weather_Condition_Severity': np.random.uniform(0, 1, n_samples),
        'Route_Risk_Level': np.random.randint(0, 11, n_samples),
        'ETA_Variation': np.random.normal(0, 2, n_samples),
        'Port_Congestion_Level': np.random.randint(0, 11, n_samples),
        'Loading_Unloading_Time': np.random.normal(2, 0.5, n_samples),
        'Delay_Probability': np.random.uniform(0, 1, n_samples)
    }
    
    return pd.DataFrame(data)

def print_separator():
    print("\n" + "="*80 + "\n")

def test_route_optimization():
    """Test the RouteOptimizer with MLflow tracking"""
    print("Starting Route Optimization Test...")
    print_separator()
    
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data()
    print(f"Generated {len(df)} samples")
    print_separator()
    
    # Initialize route optimizer
    optimizer = RouteOptimizer()
    
    # Test route optimization with MLflow tracking
    print("Testing route optimization and clustering...")
    try:
        optimization_results = optimizer.optimize_routes(df)
        
        print("\nCluster Analysis:")
        print(optimization_results['cluster_analysis'])
        
        print("\nClustering Metrics:")
        for metric, value in optimization_results['metrics'].items():
            print(f"- {metric}: {value:.4f}")
    except Exception as e:
        print(f"Error during route optimization: {str(e)}")
    print_separator()
    
    # Test delivery time analysis
    print("Testing delivery time analysis...")
    try:
        delivery_analysis = optimizer.analyze_delivery_times(df)
        
        print("\nDelivery Time Correlations:")
        for feature, correlation in delivery_analysis['correlations'].items():
            print(f"- {feature}: {correlation:.4f}")
        
        print("\nDelivery Metrics:")
        for metric, value in delivery_analysis['metrics'].items():
            print(f"- {metric}: {value:.4f}")
    except Exception as e:
        print(f"Error during delivery time analysis: {str(e)}")
    print_separator()
    
    # Test route recommendations
    print("Testing route recommendations...")
    try:
        recommendations = optimizer.recommend_optimal_routes(df)
        
        print("\nRoute Recommendations by Cluster:")
        print(recommendations['route_recommendations'])
        
        print("\nOverall Metrics:")
        for metric, value in recommendations['overall_metrics'].items():
            print(f"- {metric}: {value:.4f}")
    except Exception as e:
        print(f"Error during route recommendations: {str(e)}")
    print_separator()

if __name__ == "__main__":
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print_separator()
    test_route_optimization()