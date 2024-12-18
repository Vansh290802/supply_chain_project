# External Factors Analysis Module

## Overview
This module analyzes external factors affecting supply chain operations, including weather, traffic, and port conditions.

## Key Components

### 1. Temporal Pattern Analysis
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

### 2. Impact Analysis
Components analyzed:
- Weather impact on operations
- Traffic pattern effects
- Port congestion influence
- Combined external factors

## Implementation Details

### 1. Prediction Models
- Delay Predictor: RandomForestRegressor
- Time Deviation Predictor: GradientBoostingRegressor

### 2. Risk Score Calculation
External Risk Score = Weather (30%) + Traffic (40%) + Port (30%)

### 3. Recommendations
Generated based on:
- Weather severity thresholds
- Traffic congestion patterns
- Port operations efficiency

## Key Findings
1. Traffic Recommendations:
   - Adjust delivery schedules to avoid peak hours
2. Port Operations:
   - Consider alternative ports or implement scheduling system
3. Weather Impact:
   - Relatively stable but affects operations

## Usage
```python
from external_factors import ExternalFactorsAnalyzer

# Initialize analyzer
external_analyzer = ExternalFactorsAnalyzer()

# Analyze patterns
temporal_patterns = external_analyzer.analyze_temporal_patterns(df)
weather_analysis = external_analyzer.analyze_weather_impact(df)
traffic_analysis = external_analyzer.analyze_traffic_patterns(df)

# Get recommendations
recommendations = external_analyzer.generate_recommendations(df)
```