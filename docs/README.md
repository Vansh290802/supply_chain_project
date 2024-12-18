# Documentation Directory

This directory contains run logs and saved models from the supply chain analysis.

## Structure

- `run_logs/`: Contains detailed logs from each analysis run, including performance metrics, warnings, and error messages.
- `saved_models/`: Contains saved model artifacts and weights for each component:
  - Risk assessment models
  - Route optimization models
  - Maintenance prediction models
  - External factors analysis models
  - Inventory management models

## Usage

The logs and models in this directory are automatically generated when running the main analysis script. Each run creates a new timestamped directory under `run_logs/` to preserve the history of analysis runs.