import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class RouteOptimizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.cluster_model = KMeans(n_clusters=5, random_state=42)
        
    def optimize_routes(self, df):
        """Optimize delivery routes based on various factors"""
        # Extract relevant features for route optimization
        route_features = [
            'Traffic_Congestion_Level',
            'Weather_Condition_Severity',
            'Route_Risk_Level',
            'ETA_Variation',
            'Port_Congestion_Level'
        ]
        
        X = df[route_features]
        X_scaled = self.scaler.fit_transform(X)
        
        # Cluster routes based on characteristics
        clusters = self.cluster_model.fit_predict(X_scaled)
        
        # Analyze cluster characteristics
        df['Route_Cluster'] = clusters
        cluster_analysis = df.groupby('Route_Cluster').agg({
            'Traffic_Congestion_Level': 'mean',
            'Weather_Condition_Severity': 'mean',
            'Route_Risk_Level': 'mean',
            'ETA_Variation': 'mean',
            'Delay_Probability': 'mean'
        }).round(3)
        
        return cluster_analysis
    
    def analyze_delivery_times(self, df):
        """Analyze factors affecting delivery times"""
        delivery_correlations = df[[
            'ETA_Variation',
            'Traffic_Congestion_Level',
            'Weather_Condition_Severity',
            'Port_Congestion_Level',
            'Loading_Unloading_Time'
        ]].corr()['ETA_Variation'].sort_values(ascending=False)
        
        return delivery_correlations
    
    def recommend_optimal_routes(self, df):
        """Generate route recommendations based on analysis"""
        # Calculate risk-adjusted efficiency score
        df['Efficiency_Score'] = (
            (1 - df['Traffic_Congestion_Level']/10) * 0.3 +
            (1 - df['Weather_Condition_Severity']) * 0.2 +
            (1 - df['Route_Risk_Level']/10) * 0.3 +
            (1 - df['Delay_Probability']) * 0.2
        )
        
        route_recommendations = df.groupby('Route_Cluster').agg({
            'Efficiency_Score': 'mean',
            'ETA_Variation': 'mean',
            'Delay_Probability': 'mean'
        }).sort_values('Efficiency_Score', ascending=False)
        
        return route_recommendations