# Supply Chain Analytics Project

## Overview
This project implements a comprehensive supply chain analytics solution for Abacus Logistics, focusing on optimization of internal supply chain and logistics operations. The analysis covers various aspects including risk assessment, route optimization, maintenance prediction, external factors analysis, and inventory management.

## Project Structure
```
supply_chain_project/
│
├── src/                        # Source code
│   ├── risk_assessment/       # Risk assessment module
│   ├── route_optimization/    # Route optimization module
│   ├── maintenance_prediction/# Maintenance prediction module
│   ├── external_factors/      # External factors analysis
│   ├── inventory_management/  # Inventory management module
│   └── data_preprocessing/    # Data preprocessing utilities
│
├── data/                      # Data directory
│   ├── dynamic_supply_chain_logistics_dataset.csv    # Main dataset
│   └── sample_data.csv        # Sample dataset for testing
│
├── docs/                      # Documentation directory
│   ├── run_logs/             # Execution logs from analysis runs
│   └── saved_models/         # Saved model artifacts and weights
│
└── output_*/                  # Generated output and models
```

## Data Directory
The `data` directory contains two main files:
- `dynamic_supply_chain_logistics_dataset.csv`: The complete dataset containing all supply chain metrics from 2021-2024
- `sample_data.csv`: A smaller sample dataset useful for testing and development

## Documentation
The `docs` directory contains:
- `run_logs/`: Log files generated during analysis runs, containing detailed execution information and results
- `saved_models/`: Trained models and their weights, saved after successful analysis runs

## Key Results from Analysis

### 1. Risk Assessment
- Disruption RMSE: 0.283
- Comprehensive supplier risk analysis performed

### 2. Route Optimization
- Identified 5 route clusters with efficiency scores ranging from 0.25 to 0.55
- Best performing cluster (2) shows efficiency score of 0.554

### 3. Maintenance Metrics
- Average fuel consumption: 8.01
- High temperature incidents: 1,604
- Poor driving behavior instances: 18,101
- High fatigue incidents: 15,627

### 4. External Factors Analysis
- Identified peak traffic hours and congestion patterns
- Port congestion levels average around 6.9-7.0
- Weather severity remains relatively stable across weekdays

### 5. Inventory Management
- Safety stock level: 55,552 units
- Reorder point: 87,032 units
- Economic order quantity: 46,883 units

## Installation & Usage
```bash
# Clone the repository
git clone https://github.com/Vansh290802/supply_chain_project.git

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python main.py
```

## Component Documentation
Detailed documentation for each component can be found in their respective directories under `src/`.