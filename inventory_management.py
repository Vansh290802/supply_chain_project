import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

class InventoryManager:
    def __init__(self):
        self.demand_predictor = RandomForestRegressor(random_state=42)
        
    def analyze_inventory_levels(self, df):
        """Analyze warehouse inventory levels and patterns"""
        inventory_stats = {
            'avg_inventory': df['Warehouse_Inventory_Level'].mean(),
            'min_inventory': df['Warehouse_Inventory_Level'].min(),
            'max_inventory': df['Warehouse_Inventory_Level'].max(),
            'stockout_frequency': (df['Warehouse_Inventory_Level'] == 0).mean(),
            'inventory_turnover': df['Historical_Demand'].sum() / df['Warehouse_Inventory_Level'].mean()
        }
        
        return inventory_stats
    
    def optimize_inventory(self, df):
        """Optimize inventory levels based on demand and lead times"""
        # Calculate safety stock levels
        lead_time_mean = df['Lead_Time'].mean()
        lead_time_std = df['Lead_Time'].std()
        demand_mean = df['Historical_Demand'].mean()
        demand_std = df['Historical_Demand'].std()
        
        safety_stock = np.sqrt(
            lead_time_mean * demand_std**2 +
            demand_mean**2 * lead_time_std**2
        ) * 1.96  # 95% service level
        
        # Calculate reorder point
        reorder_point = demand_mean * lead_time_mean + safety_stock
        
        # Calculate economic order quantity (EOQ)
        holding_cost = 0.2  # 20% of item cost
        ordering_cost = 100  # Fixed cost per order
        annual_demand = demand_mean * 365
        
        eoq = np.sqrt(
            (2 * annual_demand * ordering_cost) /
            holding_cost
        )
        
        return {
            'safety_stock': safety_stock,
            'reorder_point': reorder_point,
            'economic_order_quantity': eoq
        }
    
    def predict_demand(self, df):
        """Predict future demand based on historical data"""
        # Prepare features for demand prediction
        features = [
            'Warehouse_Inventory_Level',
            'Lead_Time',
            'Supplier_Reliability_Score',
            'Historical_Demand'
        ]
        
        X = df[features]
        y = df['Historical_Demand'].shift(-1)  # Next period's demand
        
        # Remove last row with NaN target
        X = X[:-1]
        y = y[:-1]
        
        # Train model
        self.demand_predictor.fit(X, y)
        demand_forecast = self.demand_predictor.predict(X)
        
        return demand_forecast
    
    def analyze_warehouse_efficiency(self, df):
        """Analyze warehouse operations efficiency"""
        efficiency_metrics = {
            'avg_loading_time': df['Loading_Unloading_Time'].mean(),
            'equipment_availability': df['Handling_Equipment_Availability'].mean(),
            'order_fulfillment_rate': df['Order_Fulfillment_Status'].mean(),
            'inventory_accuracy': 0.95  # Assumed 95% accuracy
        }
        
        # Analyze relationship between equipment availability and loading time
        equipment_impact = df.groupby('Handling_Equipment_Availability').agg({
            'Loading_Unloading_Time': 'mean',
            'Order_Fulfillment_Status': 'mean'
        }).round(3)
        
        return efficiency_metrics, equipment_impact
    
    def generate_inventory_recommendations(self, df):
        """Generate recommendations for inventory management"""
        inventory_stats = self.analyze_inventory_levels(df)
        optimization_results = self.optimize_inventory(df)
        
        recommendations = []
        
        # Check inventory turnover
        if inventory_stats['inventory_turnover'] < 12:  # Less than monthly turnover
            recommendations.append("Consider reducing inventory levels to improve turnover")
            
        # Check stockout frequency
        if inventory_stats['stockout_frequency'] > 0.05:  # More than 5% stockout
            recommendations.append("Increase safety stock levels to reduce stockouts")
            
        # Check equipment availability
        if df['Handling_Equipment_Availability'].mean() < 0.9:  # Less than 90% availability
            recommendations.append("Improve handling equipment availability")
            
        return recommendations