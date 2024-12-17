import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
import joblib

class RiskAssessmentModel:
    def __init__(self):
        self.disruption_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.risk_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    def train(self, X_train, y_train):
        """Train the risk assessment models"""
        print("Training risk assessment models...")
        
        # Train disruption likelihood model if target exists
        if 'Disruption_Likelihood_Score' in y_train.columns:
            self.disruption_model.fit(X_train, y_train['Disruption_Likelihood_Score'])
            print("Trained disruption likelihood model")
            
        # Train risk classification model if target exists
        if 'Risk_Classification' in y_train.columns:
            self.risk_classifier.fit(X_train, y_train['Risk_Classification'])
            print("Trained risk classification model")
        
    def predict(self, X):
        """Make predictions using trained models"""
        predictions = {}
        
        # Make disruption predictions if model was trained
        if hasattr(self.disruption_model, 'feature_importances_'):
            predictions['disruption_likelihood'] = self.disruption_model.predict(X)
            
        # Make risk classification predictions if model was trained
        if hasattr(self.risk_classifier, 'feature_importances_'):
            predictions['risk_classification'] = self.risk_classifier.predict(X)
            
        return predictions
        
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        evaluation_results = {}
        predictions = self.predict(X_test)
        
        # Evaluate disruption likelihood predictions if available
        if 'disruption_likelihood' in predictions and 'Disruption_Likelihood_Score' in y_test.columns:
            disruption_rmse = np.sqrt(mean_squared_error(
                y_test['Disruption_Likelihood_Score'],
                predictions['disruption_likelihood']
            ))
            evaluation_results['disruption_rmse'] = disruption_rmse
            
        # Evaluate risk classification if available
        if 'risk_classification' in predictions and 'Risk_Classification' in y_test.columns:
            risk_report = classification_report(
                y_test['Risk_Classification'],
                predictions['risk_classification']
            )
            evaluation_results['risk_classification_report'] = risk_report
        
        return evaluation_results
        
    def analyze_supplier_risk(self, df):
        """Analyze supplier risk factors"""
        print("\nAnalyzing supplier risk factors...")
        
        # List potential metrics for supplier risk analysis
        risk_metrics = {
            'Lead_Time': 'mean',
            'Delay_Probability': 'mean',
            'Disruption_Likelihood_Score': 'mean',
            'Order_Fulfillment_Status': 'mean',
            'Delivery_Time_Deviation': 'mean'
        }
        
        # Filter metrics based on available columns
        available_metrics = {
            col: agg for col, agg in risk_metrics.items() 
            if col in df.columns
        }
        
        if not available_metrics:
            print("Warning: No standard risk metrics found in dataset")
            # Create basic risk metrics if none available
            df['Basic_Risk_Score'] = np.random.uniform(0, 1, size=len(df))
            available_metrics = {'Basic_Risk_Score': 'mean'}
        
        print(f"Using following metrics for supplier risk analysis: {list(available_metrics.keys())}")
        
        # Perform supplier risk analysis with available metrics
        if 'Supplier_Reliability_Score' in df.columns:
            supplier_risk = df.groupby('Supplier_Reliability_Score').agg(available_metrics)
        else:
            print("Warning: Supplier_Reliability_Score not found, using basic grouping")
            df['Supplier_Group'] = pd.qcut(df.index, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            supplier_risk = df.groupby('Supplier_Group').agg(available_metrics)
        
        # Round results for readability
        supplier_risk = supplier_risk.round(3)
        
        print("\nSupplier risk analysis complete")
        return supplier_risk
        
    def calculate_risk_metrics(self, df):
        """Calculate additional risk metrics"""
        risk_metrics = {}
        
        # Calculate risk metrics based on available data
        if 'Delay_Probability' in df.columns:
            risk_metrics['average_delay_probability'] = df['Delay_Probability'].mean()
            risk_metrics['high_risk_shipments'] = (df['Delay_Probability'] > 0.7).mean()
            
        if 'Order_Fulfillment_Status' in df.columns:
            risk_metrics['fulfillment_rate'] = df['Order_Fulfillment_Status'].mean()
            
        if 'Delivery_Time_Deviation' in df.columns:
            risk_metrics['avg_delivery_deviation'] = df['Delivery_Time_Deviation'].mean()
            risk_metrics['delivery_deviation_std'] = df['Delivery_Time_Deviation'].std()
            
        print("\nCalculated risk metrics:")
        for metric, value in risk_metrics.items():
            print(f"- {metric}: {value:.3f}")
            
        return risk_metrics
        
    def save_models(self, path):
        """Save trained models"""
        print(f"\nSaving risk assessment models to {path}")
        
        if hasattr(self.disruption_model, 'feature_importances_'):
            joblib.dump(self.disruption_model, f'{path}/disruption_model.pkl')
            print("Saved disruption model")
            
        if hasattr(self.risk_classifier, 'feature_importances_'):
            joblib.dump(self.risk_classifier, f'{path}/risk_classifier.pkl')
            print("Saved risk classifier")
