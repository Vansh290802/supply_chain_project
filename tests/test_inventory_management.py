import pandas as pd
import numpy as np
from src.inventory_management.inventory_management import InventoryManager
import mlflow
from datetime import datetime, timedelta

def generate_sample_data(n_samples=1000):
    """Generate sample data for testing inventory management"""
    np.random.seed(42)
    
    # Generate timestamps
    base_timestamp = datetime(2023, 1, 1)
    timestamps = [base_timestamp + timedelta(hours=x) for x in range(n_samples)]
    
    # Generate sample data
    data = {
        'Timestamp': timestamps,
        'Warehouse_Inventory_Level': np.random.normal(1000, 200, n_samples).astype(int),
        'Historical_Demand': np.random.normal(100, 20, n_samples).astype(int),
        'Lead_Time': np.random.normal(5, 1, n_samples),
        'Supplier_Reliability_Score': np.random.uniform(0.7, 1.0, n_samples),
        'Loading_Unloading_Time': np.random.normal(2, 0.5, n_samples),
        'Handling_Equipment_Availability': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
        'Order_Fulfillment_Status': np.random.choice([0, 1], n_samples, p=[0.05, 0.95])
    }
    
    # Ensure non-negative values
    data['Warehouse_Inventory_Level'] = np.maximum(0, data['Warehouse_Inventory_Level'])
    data['Historical_Demand'] = np.maximum(0, data['Historical_Demand'])
    data['Lead_Time'] = np.maximum(0, data['Lead_Time'])
    
    return pd.DataFrame(data)

def test_inventory_management():
    """Test the InventoryManager with MLflow tracking"""
    print("Starting Inventory Management Test...")
    
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data()
    print(f"Generated {len(df)} samples")
    
    # Initialize inventory manager
    inventory_manager = InventoryManager()
    
    # Test inventory level analysis
    print("\nTesting inventory level analysis...")
    inventory_stats = inventory_manager.analyze_inventory_levels(df)
    print("\nInventory Statistics:")
    for metric, value in inventory_stats.items():
        print(f"- {metric}: {value:.2f}")
    
    # Test inventory optimization with MLflow tracking
    print("\nTesting inventory optimization...")
    try:
        optimization_results = inventory_manager.optimize_inventory(df)
        print("\nOptimization Results:")
        for metric, value in optimization_results.items():
            print(f"- {metric}: {value:.2f}")
    except Exception as e:
        print(f"Error during inventory optimization: {str(e)}")
    
    # Test demand prediction with MLflow tracking
    print("\nTesting demand prediction...")
    try:
        demand_forecast, prediction_metrics = inventory_manager.predict_demand(df)
        print("\nDemand Prediction Metrics:")
        print(f"- R² Score: {prediction_metrics['r2']:.4f}")
        print(f"- RMSE: {prediction_metrics['rmse']:.4f}")
        print(f"- MAE: {prediction_metrics['mae']:.4f}")
        print(f"- CV Score: {prediction_metrics['cv_score']:.4f}")
        
        print("\nFeature Importance:")
        for feature, importance in prediction_metrics['feature_importance'].items():
            print(f"- {feature}: {importance:.4f}")
    except Exception as e:
        print(f"Error during demand prediction: {str(e)}")
    
    # Test warehouse efficiency analysis with MLflow tracking
    print("\nTesting warehouse efficiency analysis...")
    try:
        efficiency_metrics, equipment_impact = inventory_manager.analyze_warehouse_efficiency(df)
        print("\nEfficiency Metrics:")
        for metric, value in efficiency_metrics.items():
            print(f"- {metric}: {value:.4f}")
        
        print("\nEquipment Impact Analysis:")
        print(equipment_impact)
    except Exception as e:
        print(f"Error during warehouse efficiency analysis: {str(e)}")
    
    # Test recommendation generation
    print("\nTesting recommendation generation...")
    recommendations = inventory_manager.generate_inventory_recommendations(df)
    print("\nRecommendations:")
    for idx, rec in enumerate(recommendations, 1):
        print(f"{idx}. {rec}")

if __name__ == "__main__":
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    test_inventory_management()