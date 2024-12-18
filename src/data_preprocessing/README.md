# Data Preprocessing Module

## Overview
This module handles data preprocessing and standardization for the supply chain analytics project.

## Key Components

### 1. Data Loading
- Handles CSV file loading
- Performs initial data validation
- Implements error handling and logging

### 2. Column Standardization
Standardizes column names:
- Capitalizes first letters
- Converts to proper case format
- Handles special characters
- Ensures consistency across modules

### 3. Data Cleaning
Operations performed:
- Missing value handling
- Outlier detection
- Data type conversion
- Timestamp standardization

## Feature Processing

### 1. Numeric Features
- Standard scaling implementation
- Missing value imputation using mean
- Outlier handling with quantile-based approach
- Feature normalization

### 2. Categorical Features
- Label encoding for categorical variables
- Special handling for timestamps
- Preservation of temporal information
- One-hot encoding when necessary

### 3. Feature Selection
Key features processed:
```python
# Target Variables
- Disruption_Likelihood_Score
- Delay_Probability
- Risk_Classification
- Delivery_Time_Deviation

# Input Features
- Timestamp
- Vehicle_GPS_Latitude
- Vehicle_GPS_Longitude
- Fuel_Consumption_Rate
- IoT_Temperature
- Traffic_Congestion_Level
- Weather_Condition_Severity
- Port_Congestion_Level
... and others
```

## Implementation Details

### 1. Data Split Function
```python
def split_dataset(X, y):
    """
    Splits dataset into training and testing sets
    - Test size: 20%
    - Random state: 42
    - Maintains stratification where possible
    """
```

### 2. Feature Preparation
```python
def prepare_features_targets(df):
    """
    Prepares feature and target variables
    - Removes timestamp from features
    - Scales numeric features
    - Encodes categorical variables
    - Returns X, y, and column mappings
    """
```

### 3. Column Name Standardization
```python
def standardize_column_names(df):
    """
    Standardizes column names across the dataset
    - Implements consistent naming convention
    - Handles special characters
    - Maintains backward compatibility
    """
```

## Usage
```python
from data_preprocessing import load_data, prepare_features_targets, split_dataset

# Load and preprocess data
df = load_data('your_data.csv')

# Prepare features and targets
X, y, feature_columns, target_columns = prepare_features_targets(df)

# Split dataset
X_train, X_test, y_train, y_test = split_dataset(X, y)
```

## Results and Validation
- Successfully processed 32,065 records
- Handled all data types correctly
- Maintained data integrity
- Ensured consistent column naming across modules

## Error Handling
The module implements robust error handling:
- File not found exceptions
- Data format validation
- Column presence verification
- Data type checking
- Value range validation

## Dependencies
- pandas
- numpy
- scikit-learn
- Standard Python libraries

## Future Improvements
1. Add support for more data formats
2. Implement advanced outlier detection
3. Add feature selection algorithms
4. Include data validation reports
5. Add data quality scoring