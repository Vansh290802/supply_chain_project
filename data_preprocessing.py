import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import sys

def standardize_column_names(df):
    """Standardize column names to proper case and format"""
    # Define standard names for all columns
    standard_names = {
        'timestamp': 'Timestamp',
        'vehicle_gps_latitude': 'Vehicle_GPS_Latitude',
        'vehicle_gps_longitude': 'Vehicle_GPS_Longitude',
        'fuel_consumption_rate': 'Fuel_Consumption_Rate',
        'eta_variation_hours': 'ETA_Variation',
        'traffic_congestion_level': 'Traffic_Congestion_Level',
        'warehouse_inventory_level': 'Warehouse_Inventory_Level',
        'loading_unloading_time': 'Loading_Unloading_Time',
        'handling_equipment_availability': 'Handling_Equipment_Availability',
        'order_fulfillment_status': 'Order_Fulfillment_Status',
        'weather_condition_severity': 'Weather_Condition_Severity',
        'port_congestion_level': 'Port_Congestion_Level',
        'shipping_costs': 'Shipping_Costs',
        'supplier_reliability_score': 'Supplier_Reliability_Score',
        'lead_time_days': 'Lead_Time',
        'historical_demand': 'Historical_Demand',
        'iot_temperature': 'IoT_Temperature',
        'cargo_condition_status': 'Cargo_Condition_Status',
        'route_risk_level': 'Route_Risk_Level',
        'customs_clearance_time': 'Customs_Clearance_Time',
        'driver_behavior_score': 'Driver_Behavior_Score',
        'fatigue_monitoring_score': 'Fatigue_Monitoring_Score',
        'disruption_likelihood_score': 'Disruption_Likelihood_Score',
        'delay_probability': 'Delay_Probability',
        'risk_classification': 'Risk_Classification',
        'delivery_time_deviation': 'Delivery_Time_Deviation'
    }
    
    # Create a mapping for the columns that exist in the dataframe
    column_mapping = {
        col: standard_names.get(col.lower(), col) 
        for col in df.columns
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Remove any duplicate columns (case-insensitive)
    df = df.loc[:, ~df.columns.str.lower().duplicated()]
    
    return df

def load_data(file_path):
    """Load and perform initial preprocessing of the supply chain data"""
    try:
        print(f"Attempting to load file from: {file_path}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        print(f"\nInitial shape of dataset: {df.shape}")
        
        # Standardize column names
        df = standardize_column_names(df)
        print("\nStandardized columns:")
        for col in sorted(df.columns):
            print(f"- {col}")
        
        # Convert timestamp if present
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Handle categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'Timestamp':  # Skip timestamp column
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        print("\nTarget variables available:")
        target_vars = ['Disruption_Likelihood_Score', 'Delay_Probability', 
                      'Risk_Classification', 'Delivery_Time_Deviation']
        for var in target_vars:
            print(f"- {var}: {'✓' if var in df.columns else '✗'}")
        
        return df
        
    except Exception as e:
        print(f"\nError in load_data: {str(e)}")
        print(f"Python version: {sys.version}")
        print(f"Pandas version: {pd.__version__}")
        raise

def prepare_features_targets(df):
    """Prepare feature and target variables for modeling"""
    try:
        # Define target variables
        target_columns = [
            'Disruption_Likelihood_Score',
            'Delay_Probability',
            'Risk_Classification',
            'Delivery_Time_Deviation'
        ]
        
        # Remove target variables and timestamp from features
        feature_columns = [col for col in df.columns 
                         if col not in target_columns and col != 'Timestamp']
        
        print(f"\nFeature preparation:")
        print(f"- Number of features: {len(feature_columns)}")
        print(f"- Number of targets: {len(target_columns)}")
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(df[feature_columns])
        y = df[target_columns]
        
        # Print sample statistics
        print("\nFeature statistics:")
        print(f"- Mean range: [{X.mean(axis=0).min():.2f}, {X.mean(axis=0).max():.2f}]")
        print(f"- Std range: [{X.std(axis=0).min():.2f}, {X.std(axis=0).max():.2f}]")
        
        return X, y, feature_columns, target_columns
        
    except Exception as e:
        print(f"Error in prepare_features_targets: {str(e)}")
        raise

def split_dataset(X, y):
    """Split dataset into training and testing sets"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nDataset split complete:")
        print(f"- Training set: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
        print(f"- Test set: {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error in split_dataset: {str(e)}")
        raise