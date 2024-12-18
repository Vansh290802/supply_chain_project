# Risk Assessment Module

## Overview
This module implements comprehensive risk assessment for supply chain operations, focusing on supplier reliability and disruption prediction.

## Key Components

### 1. Disruption Prediction
- Implementation: RandomForestRegressor
- Performance: RMSE of 0.283
- Target Variable: Disruption_Likelihood_Score

### 2. Supplier Risk Analysis
Analyzes multiple metrics:
- Lead Time
- Delay Probability
- Disruption Likelihood
- Order Fulfillment Status
- Delivery Time Deviation

## Implementation Details

### Features Used
- Supplier_Reliability_Score
- Lead_Time
- Historical_Demand
- Weather_Condition_Severity
- Traffic_Congestion_Level
- Port_Congestion_Level

### Models
1. Disruption Model (RandomForestRegressor)
   - Predicts likelihood of disruptions
   - Uses historical patterns and current conditions

2. Risk Classifier (RandomForestClassifier)
   - Classifies risk levels
   - Categories: Low, Moderate, High Risk

## Results Analysis
From the analysis logs:
- Successfully identified high-risk suppliers
- Provided risk scores across 32,065 data points
- Demonstrated strong correlation between supplier reliability and delivery performance

## Usage
```python
from risk_assessment import RiskAssessmentModel

# Initialize model
risk_model = RiskAssessmentModel()

# Train model
risk_model.train(X_train, y_train)

# Get risk evaluation
risk_evaluation = risk_model.evaluate(X_test, y_test)

# Analyze supplier risk
supplier_risk = risk_model.analyze_supplier_risk(df)
```