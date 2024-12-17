# Supply Chain Operations Analysis

## Project Overview
This project analyzes supply chain and logistics operations data to optimize efficiency and manage risks. It processes historical data spanning from January 2021 to January 2024, covering various aspects of transportation, warehouse management, route planning, and real-time monitoring.

## Features
- **Risk Assessment**: Predictive modeling for supply chain disruptions and risk classification
- **Route Optimization**: Analysis and optimization of delivery routes
- **Maintenance Prediction**: Predictive maintenance for logistics vehicles
- **External Factors Analysis**: Impact analysis of weather, traffic, and other external factors
- **Inventory Management**: Warehouse and inventory optimization
- **Data Preprocessing**: Robust data cleaning and preparation pipeline

## Dataset Features
The dataset includes various logistics metrics such as:
- GPS coordinates
- Fuel consumption rates
- Traffic congestion levels
- Warehouse inventory levels
- Loading/unloading times
- Equipment availability
- Weather conditions
- Port congestion levels
- Supplier reliability scores
- And more...

## Project Structure
```
supply_chain_operations_big_data/
├── data_preprocessing.py     # Data cleaning and preparation
├── risk_assessment.py       # Risk analysis and prediction
├── route_optimization.py    # Route optimization logic
├── maintenance_prediction.py # Predictive maintenance
├── external_factors.py      # External factors analysis
├── inventory_management.py  # Inventory management
├── main.py                 # Main execution script
├── requirements.txt        # Project dependencies
└── Dockerfile             # Docker configuration
```

## Setup and Installation

### Using Docker
1. Build the Docker image:
```bash
docker build -t supply_chain_analytics .
```

2. Run the container:
```bash
docker run -v $(pwd):/app supply_chain_analytics
```

### Manual Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/supply_chain_operations_big_data.git
cd supply_chain_operations_big_data
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the analysis:
```bash
python main.py
```

## Input Data Format
The system expects a CSV file with the following columns:
- Timestamp
- Vehicle_GPS_Latitude
- Vehicle_GPS_Longitude
- Fuel_Consumption_Rate
- ETA_Variation
- Traffic_Congestion_Level
- Warehouse_Inventory_Level
- Loading_Unloading_Time
- Handling_Equipment_Availability
- Order_Fulfillment_Status
- Weather_Condition_Severity
- Port_Congestion_Level
- Shipping_Costs
- Supplier_Reliability_Score
- Lead_Time
- Historical_Demand
- IoT_Temperature
- Cargo_Condition_Status
- Route_Risk_Level
- Customs_Clearance_Time
- Driver_Behavior_Score
- Fatigue_Monitoring_Score

## Output
The analysis generates:
- Risk assessment reports
- Route optimization recommendations
- Maintenance schedules
- External factors impact analysis
- Inventory optimization recommendations
- Trained machine learning models

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- joblib
- seaborn
- matplotlib

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Abacus Logistics for providing the dataset
- Contributors and maintainers of the used Python libraries
- Supply chain industry experts for domain knowledge