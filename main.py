import pandas as pd
import numpy as np
# Changed from relative to absolute imports
from src.data_preprocessing import load_data, prepare_features_targets, split_dataset
from src.risk_assessment import RiskAssessmentModel
from src.route_optimization import RouteOptimizer
from src.maintenance_prediction import MaintenancePredictor
from src.external_factors import ExternalFactorsAnalyzer
from src.inventory_management import InventoryManager
import joblib
import os
from datetime import datetime
import traceback

def create_output_directory():
    """Create output directory for results and logs"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'output_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def setup_logging(output_dir):
    """Setup logging to file"""
    log_file = f'{output_dir}/analysis_log.txt'
    return log_file

def log_message(message, log_file):
    """Log message to both console and file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f'[{timestamp}] {message}'
    print(log_entry)
    with open(log_file, 'a') as f:
        f.write(log_entry + '\n')

def main():
    try:
        # Create output directory and setup logging
        output_dir = create_output_directory()
        log_file = setup_logging(output_dir)
        log_message("Starting supply chain analysis...", log_file)

        # Load and preprocess data
        log_message("Loading and preprocessing data...", log_file)
        df = load_data('data/dynamic_supply_chain_logistics_dataset.csv')  # Updated path
        X, y, feature_columns, target_columns = prepare_features_targets(df)
        X_train, X_test, y_train, y_test = split_dataset(X, y)
        
        # Risk Assessment
        log_message("Performing risk assessment...", log_file)
        risk_model = RiskAssessmentModel()
        risk_model.train(X_train, y_train)
        risk_evaluation = risk_model.evaluate(X_test, y_test)
        supplier_risk = risk_model.analyze_supplier_risk(df)
        
        log_message(f"Risk Assessment Results:", log_file)
        log_message(f"Disruption RMSE: {risk_evaluation['disruption_rmse']:.3f}", log_file)
        log_message("\nSupplier Risk Analysis:", log_file)
        log_message(str(supplier_risk), log_file)
        
        # Route Optimization
        log_message("\nOptimizing routes...", log_file)
        route_optimizer = RouteOptimizer()
        route_clusters = route_optimizer.optimize_routes(df)
        delivery_correlations = route_optimizer.analyze_delivery_times(df)
        route_recommendations = route_optimizer.recommend_optimal_routes(df)
        
        log_message("\nRoute Optimization Results:", log_file)
        log_message(str(route_recommendations), log_file)
        
        # Maintenance Prediction
        log_message("\nAnalyzing maintenance needs...", log_file)
        maintenance_predictor = MaintenancePredictor()
        vehicle_health = maintenance_predictor.analyze_vehicle_health(df)
        maintenance_schedule = maintenance_predictor.generate_maintenance_schedule(df)
        
        log_message("\nVehicle Health Metrics:", log_file)
        log_message(str(vehicle_health), log_file)
        
        # External Factors Analysis
        log_message("\nAnalyzing external factors...", log_file)
        external_analyzer = ExternalFactorsAnalyzer()
        temporal_patterns = external_analyzer.analyze_temporal_patterns(df)
        weather_analysis = external_analyzer.analyze_weather_impact(df)
        traffic_analysis = external_analyzer.analyze_traffic_patterns(df)
        port_analysis = external_analyzer.analyze_port_operations(df)
        combined_impact = external_analyzer.analyze_combined_impact(df)
        prediction_metrics = external_analyzer.predict_delivery_impact(df)
        recommendations = external_analyzer.generate_recommendations(df)
        
        log_message("\nExternal Factors Analysis Results:", log_file)
        log_message("Temporal Patterns:", log_file)
        log_message(str(temporal_patterns), log_file)
        log_message("\nRecommendations:", log_file)
        for rec in recommendations:
            log_message(f"- {rec['factor']}: {rec['recommendation']}", log_file)
        
        # Inventory Management
        log_message("\nAnalyzing inventory management...", log_file)
        inventory_manager = InventoryManager()
        inventory_stats = inventory_manager.analyze_inventory_levels(df)
        inventory_optimization = inventory_manager.optimize_inventory(df)
        inventory_recommendations = inventory_manager.generate_inventory_recommendations(df)
        
        log_message("\nInventory Optimization Results:", log_file)
        log_message(str(inventory_optimization), log_file)
        log_message("\nInventory Recommendations:", log_file)
        for rec in inventory_recommendations:
            log_message(f"- {rec}", log_file)
        
        # Save models
        log_message("\nSaving models...", log_file)
        models_dir = f'{output_dir}/models'
        os.makedirs(models_dir, exist_ok=True)
        risk_model.save_models(models_dir)
        joblib.dump(route_optimizer.cluster_model, f'{models_dir}/route_cluster_model.pkl')
        joblib.dump(maintenance_predictor.model, f'{models_dir}/maintenance_model.pkl')
        joblib.dump(external_analyzer.delay_predictor, f'{models_dir}/external_impact_model.pkl')
        joblib.dump(inventory_manager.demand_predictor, f'{models_dir}/demand_predictor.pkl')
        
        log_message(f"\nAnalysis complete! Results and models have been saved to {output_dir}", log_file)

    except Exception as e:
        log_message(f"\nERROR: An error occurred during analysis:", log_file)
        log_message(str(e), log_file)
        log_message("\nStacktrace:", log_file)
        log_message(traceback.format_exc(), log_file)
        raise

if __name__ == "__main__":
    main()