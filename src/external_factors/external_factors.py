import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

class ExternalFactorsAnalyzer:
    def __init__(self):
        self.delay_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.time_deviation_predictor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def analyze_temporal_patterns(self, df):
        """Analyze how external factors vary over time"""
        df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
        df['DayOfWeek'] = pd.to_datetime(df['Timestamp']).dt.dayofweek
        df['Month'] = pd.to_datetime(df['Timestamp']).dt.month
        
        # Analyze patterns by time period
        temporal_patterns = {
            'hourly_traffic': df.groupby('Hour')['Traffic_Congestion_Level'].mean(),
            'daily_weather': df.groupby('DayOfWeek')['Weather_Condition_Severity'].mean(),
            'monthly_port': df.groupby('Month')['Port_Congestion_Level'].mean(),
            'hourly_delays': df.groupby('Hour')['Delay_Probability'].mean()
        }
        
        return temporal_patterns
    
    def analyze_weather_impact(self, df):
        """Detailed analysis of weather impact on operations"""
        weather_analysis = {
            # Basic statistics
            'severity_stats': df.groupby('Weather_Condition_Severity').agg({
                'Delay_Probability': ['mean', 'std'],
                'ETA_Variation': ['mean', 'std'],
                'Delivery_Time_Deviation': ['mean', 'std'],
                'Fuel_Consumption_Rate': 'mean'
            }),
            
            # Impact on different metrics
            'correlation': df[['Weather_Condition_Severity', 
                             'Delay_Probability',
                             'ETA_Variation',
                             'Fuel_Consumption_Rate']].corr()['Weather_Condition_Severity'],
            
            # Weather severity thresholds
            'high_severity_impact': df[df['Weather_Condition_Severity'] > 0.7].agg({
                'Delay_Probability': 'mean',
                'ETA_Variation': 'mean',
                'Order_Fulfillment_Status': 'mean'
            })
        }
        
        return weather_analysis
    
    def analyze_traffic_patterns(self, df):
        """Comprehensive traffic pattern analysis"""
        traffic_analysis = {
            # Traffic level distribution
            'congestion_distribution': df['Traffic_Congestion_Level'].value_counts().sort_index(),
            
            # Impact by congestion level
            'level_impact': df.groupby('Traffic_Congestion_Level').agg({
                'ETA_Variation': ['mean', 'std'],
                'Fuel_Consumption_Rate': 'mean',
                'Delay_Probability': 'mean'
            }),
            
            # High congestion analysis
            'high_congestion_impact': df[df['Traffic_Congestion_Level'] > 7].agg({
                'ETA_Variation': 'mean',
                'Fuel_Consumption_Rate': 'mean',
                'Delivery_Time_Deviation': 'mean'
            })
        }
        
        # Calculate optimal departure times
        hourly_congestion = df.groupby('Hour')['Traffic_Congestion_Level'].mean()
        traffic_analysis['optimal_departure_hours'] = hourly_congestion[
            hourly_congestion < hourly_congestion.mean()
        ].index.tolist()
        
        return traffic_analysis
    
    def analyze_port_operations(self, df):
        """Analyze port congestion and its impacts"""
        port_analysis = {
            # Basic port statistics
            'congestion_stats': df.groupby('Port_Congestion_Level').agg({
                'Customs_Clearance_Time': ['mean', 'std'],
                'Loading_Unloading_Time': ['mean', 'std'],
                'Delay_Probability': 'mean'
            }),
            
            # Port efficiency metrics
            'efficiency_metrics': {
                'avg_clearance_time': df['Customs_Clearance_Time'].mean(),
                'clearance_time_std': df['Customs_Clearance_Time'].std(),
                'high_congestion_freq': (df['Port_Congestion_Level'] > 7).mean()
            },
            
            # Impact on lead times
            'lead_time_impact': df.groupby('Port_Congestion_Level')['Lead_Time'].mean()
        }
        
        return port_analysis
    
    def analyze_combined_impact(self, df):
        """Analyze combined impact of multiple external factors"""
        # Create composite risk score
        df['External_Risk_Score'] = (
            df['Weather_Condition_Severity'] * 0.3 +
            df['Traffic_Congestion_Level'] / 10 * 0.4 +
            df['Port_Congestion_Level'] / 10 * 0.3
        )
        
        combined_analysis = {
            # Risk score analysis
            'risk_score_impact': df.groupby(pd.qcut(df['External_Risk_Score'], 5)).agg({
                'Delay_Probability': 'mean',
                'ETA_Variation': 'mean',
                'Delivery_Time_Deviation': 'mean',
                'Order_Fulfillment_Status': 'mean'
            }),
            
            # Interaction effects
            'factor_interactions': df.pivot_table(
                values='Delay_Probability',
                index=pd.qcut(df['Traffic_Congestion_Level'], 3),
                columns=pd.qcut(df['Weather_Condition_Severity'], 3),
                aggfunc='mean'
            )
        }
        
        return combined_analysis
    
    def predict_delivery_impact(self, df):
        """Predict delivery impacts using machine learning"""
        # Prepare features
        features = [
            'Weather_Condition_Severity',
            'Traffic_Congestion_Level',
            'Port_Congestion_Level',
            'Fuel_Consumption_Rate',
            'Loading_Unloading_Time'
        ]
        
        X = df[features]
        y_delay = df['Delay_Probability']
        y_deviation = df['Delivery_Time_Deviation']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        self.delay_predictor.fit(X_scaled, y_delay)
        self.time_deviation_predictor.fit(X_scaled, y_deviation)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Delay_Importance': self.delay_predictor.feature_importances_,
            'Deviation_Importance': self.time_deviation_predictor.feature_importances_
        })
        
        # Model evaluation
        delay_cv_scores = cross_val_score(self.delay_predictor, X_scaled, y_delay, cv=5)
        deviation_cv_scores = cross_val_score(self.time_deviation_predictor, X_scaled, y_deviation, cv=5)
        
        prediction_metrics = {
            'feature_importance': feature_importance,
            'delay_cv_score': delay_cv_scores.mean(),
            'deviation_cv_score': deviation_cv_scores.mean()
        }
        
        return prediction_metrics
    
    def generate_recommendations(self, df):
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Weather-related recommendations
        if df['Weather_Condition_Severity'].mean() > 0.5:
            recommendations.append({
                'factor': 'Weather',
                'issue': 'High average weather severity',
                'recommendation': 'Implement weather monitoring system and alternative routing protocols'
            })
            
        # Traffic-related recommendations
        high_congestion_hours = df.groupby('Hour')['Traffic_Congestion_Level'].mean()
        peak_hours = high_congestion_hours[high_congestion_hours > high_congestion_hours.mean() + high_congestion_hours.std()].index
        recommendations.append({
            'factor': 'Traffic',
            'issue': f'Peak congestion during hours: {list(peak_hours)}',
            'recommendation': 'Adjust delivery schedules to avoid peak traffic hours'
        })
        
        # Port-related recommendations
        if df['Port_Congestion_Level'].mean() > 6:
            recommendations.append({
                'factor': 'Port Operations',
                'issue': 'High average port congestion',
                'recommendation': 'Consider alternative ports or implement port scheduling system'
            })
            
        return recommendations
    
    def save_analysis_results(self, results, path):
        """Save analysis results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'external_factors_analysis_{timestamp}.pkl'
        full_path = f'{path}/{filename}'
        
        pd.to_pickle(results, full_path)
        print(f'Analysis results saved to {full_path}')