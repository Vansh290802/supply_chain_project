# Supply Chain Analytics Project

## Overview
This project implements a comprehensive supply chain analytics solution for Abacus Logistics, focusing on optimization of internal supply chain and logistics operations. The analysis covers various aspects including risk assessment, route optimization, maintenance prediction, external factors analysis, and inventory management.

## Setup Instructions

### 1. Environment Setup
First, create a `global.env` file in the root directory with the following variables:
```env
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
MLFLOW_TRACKING_URI=https://dagshub.com/your_username/supply_chain_project.mlflow
```

To get these values:
1. Create an account on [DagsHub](https://dagshub.com/)
2. Create a new repository on DagsHub
3. Go to `Integration & Services` in your DagsHub repository
4. Find the MLflow integration section
5. Copy the provided tracking URI and credentials

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Vansh290802/supply_chain_project.git

# Install dependencies
pip install -r requirements.txt

# Install MLflow
pip install mlflow

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Load Environment Variables
```python
# The code will automatically load variables from global.env
# Make sure you've created the file as described above
```

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
├── global.env                 # Environment variables (gitignored)
└── output_*/                  # Generated output and models
```

## Model Tracking with MLflow and DagsHub
The project uses MLflow for comprehensive model tracking and monitoring, with DagsHub as the tracking server:

### Viewing Experiments
1. Go to your DagsHub repository
2. Navigate to the "Experiments" tab
3. You'll see all tracked experiments with metrics, parameters, and artifacts

### Tracked Metrics
- Model parameters and hyperparameters
- Feature importances
- Training metrics and evaluation results
- Supplier risk analysis metrics
- Model artifacts

### Running Experiments
```bash
# Run analysis with tracking
python main.py

# Experiments will automatically be logged to DagsHub
# View them in your DagsHub repository's Experiments tab
```

### Tracked Experiments
The project tracks several experiments using MLflow:

**Risk Assessment Experiment**
* Model Training
  - Model parameters (n_estimators, max_depth)
  - Feature importance scores
  - Cross-validation results
* Model Evaluation
  - RMSE scores
  - Classification reports
  - Prediction probabilities
* Supplier Risk Analysis
  - Risk metrics by supplier
  - Temporal patterns
  - Aggregated statistics
* Risk Metrics
  - Delay probabilities
  - Fulfillment rates
  - Delivery deviations

## Data Directory
The `data` directory contains two main files:
- `dynamic_supply_chain_logistics_dataset.csv`: The complete dataset containing all supply chain metrics from 2021-2024
- `sample_data.csv`: A smaller sample dataset useful for testing and development

## Documentation
The `docs` directory contains:
- `run_logs/`: Log files generated during analysis runs, containing detailed execution information and results
- `saved_models/`: Trained models and their weights, saved after successful analysis runs

## Testing
The project includes comprehensive tests for all components:

```bash
# Run all tests
python -m pytest

# Run specific MLflow logging tests
python -m pytest tests/risk_assessment/test_mlflow_logging.py
```

## Key Results from Analysis

### 1. Risk Assessment
- Disruption RMSE: 0.283
- Comprehensive supplier risk analysis performed
- All metrics tracked in MLflow and viewable on DagsHub

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

## Troubleshooting

### MLflow Connection Issues
1. Verify your `global.env` file is properly formatted and contains the correct credentials
2. Ensure you have network access to DagsHub
3. Check if your DagsHub token has the necessary permissions
4. Try running `mlflow --version` to ensure MLflow is properly installed

### Missing Experiments
If your experiments are not showing up in DagsHub, the most common cause is incorrect environment setup. Follow these steps:

1. **Critical: Environment Variables Check**
   ```bash
   # Your global.env MUST contain these three variables - experiments won't log without them!
   MLFLOW_TRACKING_USERNAME=your_dagshub_username
   MLFLOW_TRACKING_PASSWORD=your_dagshub_token
   MLFLOW_TRACKING_URI=https://dagshub.com/your_username/supply_chain_project.mlflow
   ```
   - All three variables are mandatory - missing any one will prevent experiment logging
   - Double-check there are no typos in variable names
   - Make sure the values match exactly with what DagsHub provides

2. **Common Environment Issues**
   - Don't use quotes around the values in global.env
   - Ensure there are no spaces around the '=' sign
   - Check if global.env is in the root directory of the project
   - Verify that python-dotenv is installed (`pip install python-dotenv`)

3. **Verify Connection**
   ```python
   # Add this to your code temporarily to debug:
   import os
   print(os.getenv('MLFLOW_TRACKING_URI'))  # Should show your DagsHub URI
   print(os.getenv('MLFLOW_TRACKING_USERNAME'))  # Should show your username
   ```

4. **Still Not Working?**
   - Check the console for any MLflow-related error messages
   - Verify your DagsHub repository has MLflow integration enabled
   - Try running a simple test experiment with minimal code
   - Check your DagsHub token has the necessary permissions

## Component Documentation
Detailed documentation for each component can be found in their respective directories under `src/`.