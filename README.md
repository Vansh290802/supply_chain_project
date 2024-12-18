# Supply Chain Analytics Project

## Overview
This project implements a comprehensive supply chain analytics solution for Abacus Logistics, focusing on optimization of internal supply chain and logistics operations. The analysis covers various aspects including risk assessment, route optimization, maintenance prediction, external factors analysis, and inventory management.

## Project Structure
```
supply_chain_project/
│
├── src/                        # Source code
│   ├── risk_assessment/       # Risk assessment module with MLflow tracking
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
├── tests/                     # Test directory
│   └── risk_assessment/      # Risk assessment tests including MLflow logging tests
│
├── docs/                      # Documentation directory
│   ├── run_logs/             # Execution logs from analysis runs
│   └── saved_models/         # Saved model artifacts and weights
│
├── mlruns/                    # MLflow tracking directory (gitignored)
└── output_*/                  # Generated output and models
```

## Model Tracking with MLflow
The project now includes MLflow integration for comprehensive model tracking and monitoring:

### Tracked Metrics
- Model parameters and hyperparameters
- Feature importances
- Training metrics and evaluation results
- Supplier risk analysis metrics
- Model artifacts

### MLflow Usage
```bash
# View MLflow UI
mlflow ui

# Run analysis with tracking
python main.py
```

### Tracked Experiments
- risk_assessment
  - Model training metrics
  - Evaluation results
  - Supplier risk analysis
  - Risk metrics calculation

[Rest of README content...]