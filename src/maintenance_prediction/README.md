# Maintenance Prediction Module

## Overview
This module implements predictive maintenance analytics for the vehicle fleet, focusing on preventing breakdowns and optimizing maintenance schedules.

## Key Metrics

### Current Vehicle Health Statistics
From analysis logs:
- Average fuel consumption: 8.01 L/h
- High temperature incidents: 1,604
- Poor driving behavior instances: 18,101
- High fatigue incidents: 15,627

## Implementation Details

### 1. Vehicle Stress Score Calculation
Weighted combination of:
- Fuel consumption rate (30%)
- IoT temperature (20%)
- Driver behavior score (25%)
- Fatigue monitoring (25%)

### 2. Maintenance Features
Key indicators monitored:
- Fuel_Consumption_Rate
- IoT_Temperature
- Driver_Behavior_Score
- Fatigue_Monitoring_Score
- Route_Risk_Level

### 3. Location-Based Grouping
- Uses GPS coordinates for location clustering
- Combines with time-based analysis
- Prioritizes maintenance based on both location and condition

## Predictive Model
- Implementation: GradientBoostingRegressor
- Features: Vehicle telemetry and operational data
- Output: Maintenance priority scores

## Usage
```python
from maintenance_prediction import MaintenancePredictor

# Initialize predictor
maintenance_predictor = MaintenancePredictor()

# Analyze vehicle health
vehicle_health = maintenance_predictor.analyze_vehicle_health(df)

# Generate maintenance schedule
maintenance_schedule = maintenance_predictor.generate_maintenance_schedule(df)
```

## Results Interpretation
- High number of fatigue incidents suggests need for driver management
- Temperature incidents indicate potential cooling system issues
- Poor driving behavior instances require driver training interventions