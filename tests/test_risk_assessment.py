import pandas as pd
import numpy as np
from src.risk_assessment.risk_assessment import RiskAssessmentModel
import mlflow
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

def generate_sample_data(n_samples=1000):
    """Generate sample data for testing risk assessment"""
    np.random.seed(42)
    
    # Generate timestamps
    base_timestamp = datetime(2023, 1, 1)
    timestamps = [base_timestamp + timedelta(hours=x) for x in range(n_samples)]
    
    # Generate features
    data = {
        'Timestamp': timestamps,
        'Supplier_Reliability_Score': np.random.uniform(0.5, 1.0, n_samples),
        'Lead_Time': np.random.normal(5, 1, n_samples),
        'Delay_Probability': np.random.uniform(0, 1, n_samples),
        'Order_Fulfillment_Status': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
        'Delivery_Time_Deviation': np.random.normal(0, 2, n_samples),
        'Weather_Condition_Severity': np.random.uniform(0, 1, n_samples),
        'Traffic_Congestion_Level': np.random.randint(0, 11, n_samples),
        'Port_Congestion_Level': np.random.randint(0, 11, n_samples),
        'Route_Risk_Level': np.random.randint(0, 11, n_samples),
        'IoT_Temperature': np.random.normal(60, 15, n_samples),
        'Driver_Behavior_Score': np.random.uniform(0.3, 1.0, n_samples),
        'Fatigue_Monitoring_Score': np.random.uniform(0, 1, n_samples)
    }
    
    # Generate target variables
    data['Disruption_Likelihood_Score'] = (
        0.3 * data['Weather_Condition_Severity'] +
        0.2 * data['Traffic_Congestion_Level'] / 10 +
        0.2 * data['Port_Congestion_Level'] / 10 +
        0.3 * (1 - data['Supplier_Reliability_Score']) +
        np.random.normal(0, 0.1, n_samples)
    ).clip(0, 1)
    
    # Generate risk classification based on combined factors
    risk_score = (
        data['Disruption_Likelihood_Score'] * 0.4 +
        data['Delay_Probability'] * 0.3 +
        (1 - data['Supplier_Reliability_Score']) * 0.3
    )
    data['Risk_Classification'] = pd.qcut(risk_score, q=3, labels=['Low Risk', 'Moderate Risk', 'High Risk'])
    
    return pd.DataFrame(data)

def print_separator():
    print("\n" + "="*80 + "\n")

def test_risk_assessment():
    """Test the RiskAssessmentModel with MLflow tracking"""
    print("Starting Risk Assessment Test...")
    print_separator()
    
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data()
    print(f"Generated {len(df)} samples")
    print_separator()
    
    # Prepare features and targets
    feature_columns = [
        'Weather_Condition_Severity',
        'Traffic_Congestion_Level',
        'Port_Congestion_Level',
        'Supplier_Reliability_Score',
        'Lead_Time',
        'Route_Risk_Level',
        'Driver_Behavior_Score',
        'Fatigue_Monitoring_Score'
    ]
    
    target_columns = ['Disruption_Likelihood_Score', 'Risk_Classification']
    
    X = df[feature_columns]
    y = df[target_columns]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize risk assessment model
    model = RiskAssessmentModel()
    
    # Test model training
    print("Testing model training...")
    try:
        training_results = model.train(X_train, y_train)
        
        print("\nTraining Results:")
        for model_name, results in training_results.items():
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"- Status: {results['status']}")
            print(f"- CV Score Mean: {results['cv_score_mean']:.4f}")
            print(f"- CV Score Std: {results['cv_score_std']:.4f}")
            
            print("\nFeature Importance:")
            sorted_features = sorted(
                results['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for feature, importance in sorted_features:
                print(f"- {feature}: {importance:.4f}")
    except Exception as e:
        print(f"Error during model training: {str(e)}")
    print_separator()
    
    # Test model evaluation
    print("Testing model evaluation...")
    try:
        evaluation_results = model.evaluate(X_test, y_test)
        
        if 'disruption_metrics' in evaluation_results:
            print("\nDisruption Model Metrics:")
            for metric, value in evaluation_results['disruption_metrics'].items():
                print(f"- {metric}: {value:.4f}")
                
        if 'classification_metrics' in evaluation_results:
            print("\nClassification Metrics:")
            for metric, value in evaluation_results['classification_metrics'].items():
                if metric != 'classification_report' and metric != 'avg_class_probabilities':
                    print(f"- {metric}: {value:.4f}")
                    
            print("\nClassification Report:")
            print(evaluation_results['classification_metrics']['classification_report'])
            
            print("\nAverage Class Probabilities:")
            for class_idx, prob in evaluation_results['classification_metrics']['avg_class_probabilities'].items():
                print(f"- Class {class_idx}: {prob:.4f}")
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
    print_separator()
    
    # Test supplier risk analysis
    print("Testing supplier risk analysis...")
    try:
        supplier_risk = model.analyze_supplier_risk(df)
        print("\nSupplier Risk Analysis:")
        print(supplier_risk)
    except Exception as e:
        print(f"Error during supplier risk analysis: {str(e)}")
    print_separator()
    
    # Test risk metrics calculation
    print("Testing risk metrics calculation...")
    try:
        risk_metrics = model.calculate_risk_metrics(df)
        print("\nRisk Metrics:")
        for metric, value in risk_metrics.items():
            print(f"- {metric}: {value:.4f}")
    except Exception as e:
        print(f"Error during risk metrics calculation: {str(e)}")
    print_separator()
    
    # Test model saving
    print("Testing model saving...")
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_models = model.save_models(tmp_dir)
            print("\nSaved Models:")
            for model_name, model_path in saved_models.items():
                print(f"- {model_name}: {model_path}")
    except Exception as e:
        print(f"Error during model saving: {str(e)}")
    print_separator()

if __name__ == "__main__":
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print_separator()
    test_risk_assessment()