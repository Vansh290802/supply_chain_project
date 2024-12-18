# External Factors Analysis Module

## Overview
This module analyzes external factors affecting supply chain operations, including weather, traffic, and port conditions. The module now includes MLflow tracking for model performance and parameter optimization.

## Key Components

### 1. MLflow Tracking
The module tracks two main models:

#### Delay Predictor (RandomForestRegressor)
Tracked metrics and parameters:
- Model parameters:
  - n_estimators
  - max_depth
  - random_state
- Performance metrics:
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² Score
  - Cross-validation scores (mean and std)
- Feature importance for all input features

#### Time Deviation Predictor (GradientBoostingRegressor)
Tracked metrics and parameters:
- Model parameters:
  - n_estimators
  - learning_rate
  - random_state
- Performance metrics:
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² Score
  - Cross-validation scores (mean and std)
- Feature importance for all input features

### 2. Temporal Pattern Analysis
From analysis logs:
#### Traffic Patterns
- Peak hours identified (hours 0, 8, and 20)
- Average congestion levels: 4.8-5.2
#### Weather Patterns
- Relatively stable across weekdays (0.49-0.51)
#### Port Congestion
- Monthly averages: 6.9-7.1
- Highest in June (7.09)
#### Delay Probabilities
- Range: 0.68-0.72
- Peak at hour 13 (0.716)

### 3. Impact Analysis
Components analyzed:
- Weather impact on operations
- Traffic pattern effects
- Port congestion influence
- Combined external factors

## Implementation Details

### 1. Prediction Models with MLflow Integration
- Delay Predictor: RandomForestRegressor
  - Experiment name: "external_factors_analysis"
  - Run name: "delay_predictor"
- Time Deviation Predictor: GradientBoostingRegressor
  - Experiment name: "external_factors_analysis"
  - Run name: "time_deviation_predictor"

### 2. Risk Score Calculation
External Risk Score = Weather (30%) + Traffic (40%) + Port (30%)

## Usage
```python
from external_factors import ExternalFactorsAnalyzer

# Initialize analyzer
external_analyzer = ExternalFactorsAnalyzer()

# Analyze patterns
temporal_patterns = external_analyzer.analyze_temporal_patterns(df)
weather_analysis = external_analyzer.analyze_weather_impact(df)
traffic_analysis = external_analyzer.analyze_traffic_patterns(df)

# Run predictions with MLflow tracking
prediction_metrics = external_analyzer.predict_delivery_impact(df)

# Access tracked metrics
print(f"Delay Predictor R² Score: {prediction_metrics['delay_metrics']['r2']}")
print(f"Time Deviation RMSE: {prediction_metrics['deviation_metrics']['rmse']}")

# Get recommendations
recommendations = external_analyzer.generate_recommendations(df)
```

## Viewing MLflow Results
To view the tracked experiments and metrics:
1. Start the MLflow UI:
   ```bash
   mlflow ui
   ```
2. Open http://localhost:5000 in your browser
3. Navigate to the "external_factors_analysis" experiment
4. View detailed metrics, parameters, and model artifacts for each run

## Dependencies
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- mlflow