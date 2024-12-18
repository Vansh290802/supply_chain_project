import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import mlflow
import mlflow.sklearn

class RouteOptimizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.cluster_model = KMeans(
            n_clusters=5,
            random_state=42,
            max_iter=300,
            n_init=10
        )
        
    def optimize_routes(self, df):
        """Optimize delivery routes based on various factors"""
        mlflow.set_experiment("route_optimization")
        
        with mlflow.start_run(run_name="route_clustering"):
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
            
            # Log parameters
            mlflow.log_params({
                "model_type": "KMeans",
                "n_clusters": self.cluster_model.n_clusters,
                "max_iter": self.cluster_model.max_iter,
                "n_init": self.cluster_model.n_init,
                "random_state": self.cluster_model.random_state,
                "features": route_features
            })
            
            # Cluster routes based on characteristics
            clusters = self.cluster_model.fit_predict(X_scaled)
            
            # Calculate clustering metrics
            silhouette = silhouette_score(X_scaled, clusters)
            calinski = calinski_harabasz_score(X_scaled, clusters)
            inertia = self.cluster_model.inertia_
            
            # Log metrics
            mlflow.log_metrics({
                "silhouette_score": silhouette,
                "calinski_harabasz_score": calinski,
                "inertia": inertia
            })
            
            # Analyze cluster characteristics
            df['Route_Cluster'] = clusters
            cluster_analysis = df.groupby('Route_Cluster').agg({
                'Traffic_Congestion_Level': 'mean',
                'Weather_Condition_Severity': 'mean',
                'Route_Risk_Level': 'mean',
                'ETA_Variation': 'mean',
                'Delay_Probability': 'mean'
            }).round(3)
            
            # Log cluster characteristics
            for cluster in cluster_analysis.index:
                for metric in cluster_analysis.columns:
                    mlflow.log_metric(
                        f"cluster_{cluster}_{metric}",
                        cluster_analysis.loc[cluster, metric]
                    )
            
            # Log model
            mlflow.sklearn.log_model(self.cluster_model, "route_cluster_model")
            
            analysis_results = {
                'cluster_analysis': cluster_analysis,
                'metrics': {
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski,
                    'inertia': inertia
                }
            }
        
        return analysis_results
    
    def analyze_delivery_times(self, df):
        """Analyze factors affecting delivery times"""
        mlflow.set_experiment("route_optimization")
        
        with mlflow.start_run(run_name="delivery_time_analysis"):
            delivery_correlations = df[[
                'ETA_Variation',
                'Traffic_Congestion_Level',
                'Weather_Condition_Severity',
                'Port_Congestion_Level',
                'Loading_Unloading_Time'
            ]].corr()['ETA_Variation'].sort_values(ascending=False)
            
            # Log correlations as metrics
            for feature in delivery_correlations.index:
                if feature != 'ETA_Variation':  # Skip self-correlation
                    mlflow.log_metric(
                        f"correlation_{feature}",
                        delivery_correlations[feature]
                    )
            
            # Calculate additional delivery metrics
            delivery_metrics = {
                'mean_eta_variation': df['ETA_Variation'].mean(),
                'std_eta_variation': df['ETA_Variation'].std(),
                'on_time_delivery_rate': (df['ETA_Variation'] <= 0).mean()
            }
            
            # Log delivery metrics
            mlflow.log_metrics(delivery_metrics)
            
            analysis_results = {
                'correlations': delivery_correlations,
                'metrics': delivery_metrics
            }
            
        return analysis_results
    
    def recommend_optimal_routes(self, df):
        """Generate route recommendations based on analysis"""
        mlflow.set_experiment("route_optimization")
        
        with mlflow.start_run(run_name="route_recommendations"):
            # Log weighting parameters
            weights = {
                'traffic_weight': 0.3,
                'weather_weight': 0.2,
                'risk_weight': 0.3,
                'delay_weight': 0.2
            }
            mlflow.log_params(weights)
            
            # Calculate risk-adjusted efficiency score
            df['Efficiency_Score'] = (
                (1 - df['Traffic_Congestion_Level']/10) * weights['traffic_weight'] +
                (1 - df['Weather_Condition_Severity']) * weights['weather_weight'] +
                (1 - df['Route_Risk_Level']/10) * weights['risk_weight'] +
                (1 - df['Delay_Probability']) * weights['delay_weight']
            )
            
            route_recommendations = df.groupby('Route_Cluster').agg({
                'Efficiency_Score': 'mean',
                'ETA_Variation': 'mean',
                'Delay_Probability': 'mean'
            }).sort_values('Efficiency_Score', ascending=False)
            
            # Log metrics for each cluster
            for cluster in route_recommendations.index:
                for metric in route_recommendations.columns:
                    mlflow.log_metric(
                        f"cluster_{cluster}_{metric}",
                        route_recommendations.loc[cluster, metric]
                    )
            
            # Calculate overall metrics
            overall_metrics = {
                'avg_efficiency_score': df['Efficiency_Score'].mean(),
                'best_cluster_efficiency': route_recommendations['Efficiency_Score'].max(),
                'worst_cluster_efficiency': route_recommendations['Efficiency_Score'].min(),
                'efficiency_score_std': df['Efficiency_Score'].std()
            }
            
            # Log overall metrics
            mlflow.log_metrics(overall_metrics)
            
            recommendations = {
                'route_recommendations': route_recommendations,
                'overall_metrics': overall_metrics
            }
            
        return recommendations