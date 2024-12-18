import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
import joblib
import mlflow
import mlflow.sklearn
import re

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

    def _sanitize_metric_name(self, name):
        """Sanitize metric names for MLflow compatibility"""
        # Replace % with 'pct'
        name = name.replace('%', 'pct')
        # Replace any other invalid characters with underscore
        name = re.sub(r'[^a-zA-Z0-9_\-. /]', '_', name)
        return name
        
    def train(self, X_train, y_train):
        """Train the risk assessment models"""
        mlflow.set_experiment("risk_assessment")
        print("Training risk assessment models...")
        
        # Handle both DataFrame and numpy array inputs
        feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Train disruption likelihood model if target exists
        if isinstance(y_train, pd.DataFrame) and 'Disruption_Likelihood_Score' in y_train.columns:
            with mlflow.start_run(run_name="disruption_model_training"):
                # Log parameters
                mlflow.log_params({
                    "model_type": "RandomForestRegressor",
                    "n_estimators": self.disruption_model.n_estimators,
                    "max_depth": self.disruption_model.max_depth,
                    "features": feature_names
                })
                
                # Train model
                self.disruption_model.fit(X_train, y_train['Disruption_Likelihood_Score'])
                print("Trained disruption likelihood model")
                
                # Log feature importances
                for idx, feature in enumerate(feature_names):
                    safe_feature_name = self._sanitize_metric_name(f"feature_importance_{feature}")
                    mlflow.log_metric(safe_feature_name, 
                                    self.disruption_model.feature_importances_[idx])
            
        # Train risk classification model if target exists
        if isinstance(y_train, pd.DataFrame) and 'Risk_Classification' in y_train.columns:
            with mlflow.start_run(run_name="risk_classifier_training"):
                # Log parameters
                mlflow.log_params({
                    "model_type": "RandomForestClassifier",
                    "n_estimators": self.risk_classifier.n_estimators,
                    "max_depth": self.risk_classifier.max_depth,
                    "features": feature_names
                })
                
                # Train model
                self.risk_classifier.fit(X_train, y_train['Risk_Classification'])
                print("Trained risk classification model")
                
                # Log feature importances
                for idx, feature in enumerate(feature_names):
                    safe_feature_name = self._sanitize_metric_name(f"feature_importance_{feature}")
                    mlflow.log_metric(safe_feature_name, 
                                    self.risk_classifier.feature_importances_[idx])
        
    def predict(self, X):
        """Make predictions using trained models"""
        predictions = {}
        
        # Make disruption predictions if model was trained
        if hasattr(self.disruption_model, 'feature_importances_'):
            predictions['disruption_likelihood'] = self.disruption_model.predict(X)
            
        # Make risk classification predictions if model was trained
        if hasattr(self.risk_classifier, 'feature_importances_'):
            predictions['risk_classification'] = self.risk_classifier.predict(X)
            predictions['risk_probabilities'] = self.risk_classifier.predict_proba(X)
            
        return predictions
        
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        mlflow.set_experiment("risk_assessment")
        evaluation_results = {}
        predictions = self.predict(X_test)
        
        # Evaluate disruption likelihood predictions if available
        if 'disruption_likelihood' in predictions and 'Disruption_Likelihood_Score' in y_test.columns:
            with mlflow.start_run(run_name="disruption_model_evaluation"):
                disruption_rmse = np.sqrt(mean_squared_error(
                    y_test['Disruption_Likelihood_Score'],
                    predictions['disruption_likelihood']
                ))
                evaluation_results['disruption_rmse'] = disruption_rmse
                
                # Log metrics
                mlflow.log_metric("rmse", disruption_rmse)
            
        # Evaluate risk classification if available
        if 'risk_classification' in predictions and 'Risk_Classification' in y_test.columns:
            with mlflow.start_run(run_name="risk_classifier_evaluation"):
                risk_report = classification_report(
                    y_test['Risk_Classification'],
                    predictions['risk_classification']
                )
                evaluation_results['risk_classification_report'] = risk_report
                
                # Log the report as artifact
                mlflow.log_text(risk_report, "classification_report.txt")
        
        return evaluation_results
        
    def analyze_supplier_risk(self, df):
        """Analyze supplier risk factors"""
        mlflow.set_experiment("risk_assessment")
        print("\nAnalyzing supplier risk factors...")
        
        with mlflow.start_run(run_name="supplier_risk_analysis"):
            # List potential metrics for supplier risk analysis
            risk_metrics = {
                'Lead_Time': 'mean',
                'Delay_Probability': 'mean',
                'Disruption_Likelihood_Score': 'mean',
                'Order_Fulfillment_Status': 'mean',
                'Delivery_Time_Deviation': 'mean'
            }
            
            # Log analyzed metrics
            mlflow.log_param("analyzed_metrics", list(risk_metrics.keys()))
            
            # Filter metrics based on available columns
            available_metrics = {
                col: agg for col, agg in risk_metrics.items() 
                if col in df.columns
            }
            
            if not available_metrics:
                print("Warning: No standard risk metrics found in dataset")
                df['Basic_Risk_Score'] = np.random.uniform(0, 1, size=len(df))
                available_metrics = {'Basic_Risk_Score': 'mean'}
                mlflow.log_param("risk_score_type", "basic")
            else:
                mlflow.log_param("risk_score_type", "comprehensive")
            
            print(f"Using following metrics for supplier risk analysis: {list(available_metrics.keys())}")
            
            # Perform supplier risk analysis with available metrics
            if 'Supplier_Reliability_Score' in df.columns:
                supplier_risk = df.groupby('Supplier_Reliability_Score').agg(available_metrics)
                mlflow.log_param("grouping_method", "Supplier_Reliability_Score")
            else:
                print("Warning: Supplier_Reliability_Score not found, using basic grouping")
                df['Supplier_Group'] = pd.qcut(df.index, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                supplier_risk = df.groupby('Supplier_Group').agg(available_metrics)
                mlflow.log_param("grouping_method", "quintile_based_groups")
            
            # Round results for readability
            supplier_risk = supplier_risk.round(3)
            
            # Log summary statistics
            for metric in available_metrics.keys():
                metric_stats = supplier_risk[metric].describe()
                for stat_name, value in metric_stats.items():
                    # Sanitize metric name
                    safe_metric_name = self._sanitize_metric_name(f"{metric}_{stat_name}")
                    mlflow.log_metric(safe_metric_name, value)
            
            print("\nSupplier risk analysis complete")
            return supplier_risk
        
    def calculate_risk_metrics(self, df):
        """Calculate additional risk metrics"""
        mlflow.set_experiment("risk_assessment")
        risk_metrics = {}
        
        with mlflow.start_run(run_name="risk_metrics_calculation"):
            # Calculate risk metrics based on available data
            if 'Delay_Probability' in df.columns:
                risk_metrics['average_delay_probability'] = df['Delay_Probability'].mean()
                risk_metrics['high_risk_shipments'] = (df['Delay_Probability'] > 0.7).mean()
                
            if 'Order_Fulfillment_Status' in df.columns:
                risk_metrics['fulfillment_rate'] = df['Order_Fulfillment_Status'].mean()
                
            if 'Delivery_Time_Deviation' in df.columns:
                risk_metrics['avg_delivery_deviation'] = df['Delivery_Time_Deviation'].mean()
                risk_metrics['delivery_deviation_std'] = df['Delivery_Time_Deviation'].std()
            
            # Log sanitized metrics to MLflow
            safe_metrics = {self._sanitize_metric_name(k): v for k, v in risk_metrics.items()}
            mlflow.log_metrics(safe_metrics)
            
            print("\nCalculated risk metrics:")
            for metric, value in risk_metrics.items():
                print(f"- {metric}: {value:.3f}")
                
            return risk_metrics
        
    def save_models(self, path):
        """Save trained models"""
        mlflow.set_experiment("risk_assessment")
        print(f"\nSaving risk assessment models to {path}")
        
        with mlflow.start_run(run_name="model_saving"):
            saved_models = []
            
            if hasattr(self.disruption_model, 'feature_importances_'):
                model_path = f'{path}/disruption_model.pkl'
                joblib.dump(self.disruption_model, model_path)
                mlflow.log_artifact(model_path)
                saved_models.append("disruption_model")
                print("Saved disruption model")
                
            if hasattr(self.risk_classifier, 'feature_importances_'):
                model_path = f'{path}/risk_classifier.pkl'
                joblib.dump(self.risk_classifier, model_path)
                mlflow.log_artifact(model_path)
                saved_models.append("risk_classifier")
                print("Saved risk classifier")
            
            mlflow.log_param("saved_models", saved_models)