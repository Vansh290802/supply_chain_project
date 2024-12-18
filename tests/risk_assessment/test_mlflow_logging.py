import unittest
import pandas as pd
import numpy as np
import mlflow
from src.risk_assessment.risk_assessment import RiskAssessmentModel

class TestMLflowLogging(unittest.TestCase):
    def setUp(self):
        # Create sample data
        self.X_train = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
        })
        self.y_train = pd.DataFrame({
            'Disruption_Likelihood_Score': np.random.rand(100),
            'Risk_Classification': np.random.choice([0, 1], 100)
        })
        
        # Initialize model
        self.model = RiskAssessmentModel()
        
        # Set test experiment
        mlflow.set_experiment("test_risk_assessment")

    def test_training_logs_parameters(self):
        """Test if model training logs parameters correctly"""
        self.model.train(self.X_train, self.y_train)
        
        # Get latest run
        runs = mlflow.search_runs(experiment_names=["test_risk_assessment"])
        latest_run = runs.iloc[0]
        
        # Check parameters were logged
        self.assertIn("model_type", latest_run.keys())
        self.assertIn("n_estimators", latest_run.keys())
        self.assertIn("max_depth", latest_run.keys())
        self.assertIn("features", latest_run.keys())

    def test_evaluation_logs_metrics(self):
        """Test if model evaluation logs metrics correctly"""
        # Train model first
        self.model.train(self.X_train, self.y_train)
        
        # Evaluate model
        self.model.evaluate(self.X_train, self.y_train)
        
        # Get latest run
        runs = mlflow.search_runs(experiment_names=["test_risk_assessment"])
        latest_run = runs.iloc[0]
        
        # Check metrics were logged
        self.assertIn("rmse", latest_run.keys())

    def test_supplier_risk_logs(self):
        """Test if supplier risk analysis logs metrics correctly"""
        # Create sample DataFrame for supplier risk analysis
        df = pd.DataFrame({
            'Lead_Time': np.random.rand(100),
            'Delay_Probability': np.random.rand(100),
            'Supplier_Reliability_Score': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Run analysis
        self.model.analyze_supplier_risk(df)
        
        # Get latest run
        runs = mlflow.search_runs(experiment_names=["test_risk_assessment"])
        latest_run = runs.iloc[0]
        
        # Check if summary statistics were logged
        self.assertTrue(any('Lead_Time' in key for key in latest_run.keys()))
        self.assertTrue(any('Delay_Probability' in key for key in latest_run.keys()))

    def test_metric_name_sanitization(self):
        """Test if metric names are properly sanitized"""
        # Test sanitization directly
        test_cases = {
            'metric%': 'metricpct',
            'metric@123': 'metric_123',
            'metric name': 'metric name',
            'metric/path': 'metric/path',
            'metric-value': 'metric-value',
        }
        
        for input_name, expected in test_cases.items():
            sanitized = self.model._sanitize_metric_name(input_name)
            self.assertEqual(sanitized, expected)