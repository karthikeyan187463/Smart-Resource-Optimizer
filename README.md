# Smart Resource Optimizer (SRO) - Detailed Documentation

A comprehensive AI-powered cloud resource optimization system that predicts CPU requirements, recommends optimal instance configurations, and schedules workloads for maximum cost efficiency using spot pricing strategies.

## Table of Contents

1. [Overview](#overview)
2. [Core Components Deep Dive](#core-components-deep-dive)
   - [SmartResourceOptimizer Class (`sro.py`)](#1-smartresourceoptimizer-class-sropy)
     - [Initialization and Configuration](#initialization-and-configuration)
     - [Role Mapping System](#role-mapping-system)
     - [CSV Pricing Data Loading](#csv-pricing-data-loading)
     - [CPU Prediction with Error Handling](#cpu-prediction-with-error-handling)
     - [Instance Selection Algorithm](#instance-selection-algorithm)
     - [Spot Pricing Simulation](#spot-pricing-simulation)
   - [Prediction Server (`predictor.py`)](#2-prediction-server-predictorpy)
     - [Model Loading and Initialization](#model-loading-and-initialization)
     - [Data Sanitization](#data-sanitization)
     - [Feature Engineering Pipeline](#feature-engineering-pipeline)
     - [Prediction Endpoint](#prediction-endpoint)
   - [Streamlit Web Application (`streamlit_app.py`)](#3-streamlit-web-application-streamlit_apppy)
     - [Template Loading and Management](#template-loading-and-management)
     - [Session State Management](#session-state-management)
     - [Template Selection Logic](#template-selection-logic)
     - [Form Validation and Processing](#form-validation-and-processing)
     - [Results Display and Visualization](#results-display-and-visualization)
     - [Cost Comparison Visualization](#cost-comparison-visualization)
     - [Utilization Gauges](#utilization-gauges)
     - [Scheduling Visualization](#scheduling-visualization)
   - [Instance Pricing Fetcher (`fetch_instance_pricing.py`)](#4-instance-pricing-fetcher-fetch_instance_pricingpy)
     - [Region Mapping](#region-mapping)
     - [Pricing API Integration](#pricing-api-integration)
     - [Data Processing and Validation](#data-processing-and-validation)
3. [Workload Templates Structure](#workload-templates-structure)
4. [System Workflow](#system-workflow)
5. [Key Benefits](#key-benefits)

## Overview

The Smart Resource Optimizer helps organizations reduce cloud computing costs by intelligently matching workload requirements with optimal instance configurations and timing. The system combines machine learning predictions with real-time pricing data to deliver significant cost savings while maintaining performance requirements.

## Core Components Deep Dive

### 1. SmartResourceOptimizer Class (`sro.py`)

The heart of the optimization system that handles prediction, instance selection, and workload scheduling.

#### Initialization and Configuration

```python
def __init__(self, pred_endpoint="http://127.0.0.1:5000/predict", timeout=10, csv_file="./data/csv/aws_pricing.csv"):
    self.pred_endpoint = pred_endpoint
    self.timeout = timeout
    self.csv_file = csv_file
    self.instance_pricing = self._load_csv_pricing()
```
**Purpose**: Sets up the optimizer with prediction server endpoint, request timeout, and pricing data file path. The `_load_csv_pricing()` method loads instance pricing data during initialization.

#### Role Mapping System

```python
ROLE_MAPPING = {
    "ML_TRAIN": NodeRole.HN,
    "ML_INFERENCE": NodeRole.CN,
    "ETL": NodeRole.CN
}

ROLE_DISPLAY_MAPPING = {
    "Machine Learning Training": "ML_TRAIN",
    "Machine Learning Inference": "ML_INFERENCE", 
    "Data Processing (ETL)": "ETL"
}
```
**Purpose**: Maps user-friendly workload display names to internal role classifications. ML training requires high-performance nodes (HN), while inference and ETL use compute nodes (CN).

```python
def map_role(self, workload_config):
    original_role = workload_config.get("role", "unknown")
    mapped_role = ROLE_MAPPING.get(original_role, NodeRole.CN)
    return mapped_role.value
```
**Purpose**: Converts workload roles to node types for ML model prediction. Defaults to compute node (CN) for unknown roles.

#### CSV Pricing Data Loading

```python
pricing_data = {}
with open(self.csv_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        cpu_count = int(row['cpu_count'])
        hourly_cost = float(row['hourly_cost'])
        memory_gb = int(row['memory_gb'])
        instance_name = row['instance_name']
        
        if cpu_count not in pricing_data or hourly_cost < pricing_data[cpu_count][0]:
            pricing_data[cpu_count] = (hourly_cost, memory_gb, instance_name)
```
**Purpose**: Loads pricing data and keeps only the cheapest instance for each CPU count. This optimization reduces search space and ensures cost-effective recommendations.

#### CPU Prediction with Error Handling

```python
payload = {
    "gpu_request": workload_config.get("gpu_request", 0),
    "disk_request": workload_config.get("disk_request", 0), 
    "memory_request": workload_config.get("memory_request", 0),
    "max_instance_per_node": workload_config.get("max_instance_per_node", 1),
    "role": mapped_role,
    "app_name": workload_config.get("app_name", "unknown")
}
```
**Purpose**: Constructs API payload for ML prediction server, ensuring all required fields have default values.

```python
for key, value in payload.items():
    if pd.isna(value) or value is None:
        if key in ["gpu_request", "disk_request", "memory_request", "max_instance_per_node"]:
            payload[key] = 0
        else:
            payload[key] = "UNKNOWN"
```
**Purpose**: Sanitizes payload data by replacing null/NaN values with appropriate defaults. Numeric fields get 0, categorical fields get "UNKNOWN".

```python
response = requests.post(
    self.pred_endpoint, 
    json=payload, 
    timeout=self.timeout
)

if response.status_code == 200:
    try:
        result = response.json()
        predicted_cpu = result.get('cpu_pred', 1)
        return max(1, round(predicted_cpu))
    except json.JSONDecodeError:
        print(f"Invalid response: {response.text}")
        return 1
```
**Purpose**: Makes HTTP request to prediction server with timeout protection. Ensures minimum 1 CPU core and handles JSON parsing errors gracefully.

#### Instance Selection Algorithm

```python
best_instance = None
min_cost = float('inf')

for instance in all_instances:
    cpu_count = instance['cpu_count']
    memory_gb = instance['memory_gb']
    hourly_cost = instance['hourly_cost']
    instance_name = instance['instance_name']
    memory_mb = memory_gb * 1024

    # Check whether the instance can handle the workload
    if cpu_count >= predicted_cpu and memory_mb >= memory_needed:
        if hourly_cost < min_cost:
            min_cost = hourly_cost
            best_instance = {
                'instance_name': instance_name,
                'cpu_count': cpu_count,
                'memory_gb': memory_gb,
                'hourly_cost': hourly_cost,
                'cpu_utilization': (predicted_cpu / cpu_count) * 100,
                'memory_utilization': (memory_needed / memory_mb) * 100
            }
```
**Purpose**: 
- `if cpu_count >= predicted_cpu and memory_mb >= memory_needed:` - Validates that instance meets minimum CPU and memory requirements
- `if hourly_cost < min_cost:` - Ensures we select the most cost-effective option among suitable instances
- `'cpu_utilization': (predicted_cpu / cpu_count) * 100` - Calculates CPU utilization percentage for performance monitoring
- `'memory_utilization': (memory_needed / memory_mb) * 100` - Calculates memory utilization for efficiency analysis

#### Spot Pricing Simulation

```python
# Hourly cost multipliers simulating on demand instance pricing patterns
hourly_multipliers = [
    0.3, 0.25, 0.2, 0.2, 0.25, 0.4,  # 12AM-6AM: Off-peak, lowest prices
    0.6, 0.8, 1.0, 1.0, 0.9, 0.8,    # 6AM-12PM: Morning ramp-up
    0.9, 1.0, 1.0, 0.9, 0.8, 0.7,    # 12PM-6PM: Peak hours
    0.6, 0.5, 0.4, 0.35, 0.3, 0.3    # 6PM-12AM: Evening decline
]
```
**Purpose**: Simulates realistic 24-hour spot pricing patterns. Early morning hours (12AM-6AM) have lowest multipliers (20-30% of base cost), while business hours have higher multipliers (80-100%).

```python
for start_hour in range(math.ceil(time_horizon_hours - execution_duration + 1)):
    total_cost = 0
    for hour in range(math.ceil(execution_duration)):
        hour_idx = (start_hour + hour) % 24
        multiplier = hourly_multipliers[hour_idx]
        spot_cost = base_cost * multiplier
        total_cost += spot_cost
    
    if total_cost < min_total_cost:
        min_total_cost = total_cost
        best_start_hour = start_hour
```
**Purpose**: 
- `range(math.ceil(time_horizon_hours - execution_duration + 1))` - Ensures workload can complete within time horizon
- `hour_idx = (start_hour + hour) % 24` - Handles day wraparound for executions spanning midnight
- `spot_cost = base_cost * multiplier` - Applies hourly pricing multiplier to base instance cost
- Finds optimal start time that minimizes total execution cost

```python
on_demand_cost = base_cost * execution_duration
savings = on_demand_cost - min_total_cost
start_time = base_time + timedelta(hours=best_start_hour)
end_time = start_time + timedelta(hours=execution_duration)
```
**Purpose**: 
- `on_demand_cost = base_cost * execution_duration` - Calculates traditional on-demand pricing for comparison
- `savings = on_demand_cost - min_total_cost` - Determines cost savings from optimal scheduling
- Creates proper datetime objects for scheduled execution window

### 2. Prediction Server (`predictor.py`)

Flask-based API server providing CPU resource predictions using pre-trained ML models.

#### Model Loading and Initialization

```python
with open(BASE / "./data/ml/model.pkl", "rb") as f:
    model = pickle.load(f)
with open(BASE / "./data/ml/encoder_role.pkl", "rb") as f:
    enc_role = pickle.load(f)
with open(BASE / "./data/ml/encoder_app.pkl", "rb") as f:
    enc_app = pickle.load(f)
with open(BASE / "./data/ml/columns_order.pkl", "rb") as f:
    columns_order = pickle.load(f)
```
**Purpose**: Loads pre-trained ML artifacts including the prediction model, categorical encoders, and column ordering for consistent feature alignment.

#### Data Sanitization

```python
def safe_number(v):
    try:
        if v is None:
            return 0.0
        if isinstance(v, (int, float, np.integer, np.floating)):
            if np.isnan(v) or np.isinf(v):
                return 0.0
            return float(v)
        return float(v)
    except Exception:
        return 0.0
```
**Purpose**: Robust number conversion that handles None, NaN, infinity, and invalid values by defaulting to 0.0. Prevents model prediction errors from bad input data.

#### Feature Engineering Pipeline

```python
# Ensure numeric columns exist & sanitize
for col in NUMERIC_FEATS:
    if col not in df.columns:
        df[col] = 0.0
    df[col] = df[col].apply(safe_number)

# Ensure categorical columns exist & sanitize
for c in CATEGORICAL_FEATS:
    if c not in df.columns:
        df[c] = "UNKNOWN"
    else:
        df[c] = df[c].fillna("UNKNOWN").astype(str)
```
**Purpose**: Ensures all required features exist in the input dataframe. Missing numeric features get 0.0, missing categorical features get "UNKNOWN" placeholder.

```python
# One-hot encoding
role_ohe = enc_role.transform(df[["role"]])
app_ohe = enc_app.transform(df[["app_name"]])

# Align with training columns
X_final = X_final.reindex(columns=columns_order, fill_value=0)
```
**Purpose**: 
- Applies one-hot encoding to categorical features using pre-trained encoders
- `X_final.reindex(columns=columns_order, fill_value=0)` ensures feature columns match training data order exactly

#### Prediction Endpoint

```python
@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        X = preprocess_input(payload)
        preds = model.predict(X)
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
        
        if isinstance(payload, dict):
            return jsonify({"cpu_pred": float(preds[0])})
        else:
            return jsonify({"cpu_pred": [float(x) for x in preds]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
```
**Purpose**: 
- `request.get_json(force=True)` - Forces JSON parsing even with incorrect content-type
- `np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)` - Sanitizes model predictions by replacing invalid values
- Handles both single predictions (dict input) and batch predictions (list input)

### 3. Streamlit Web Application (`streamlit_app.py`)

Interactive web interface with comprehensive workload optimization capabilities.

#### Template Loading and Management

```python
def load_workloads(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return []

    try:
        with open(file_path, "r") as f:
            workloads = json.load(f)
            return workloads
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []
```
**Purpose**: Safely loads workload templates with error handling for missing files and invalid JSON.

#### Session State Management

```python
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = SmartResourceOptimizer()
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'workload_config' not in st.session_state:
    st.session_state.workload_config = {}
```
**Purpose**: Initializes Streamlit session state to maintain optimizer instance, prediction status, and workload configuration across user interactions.

#### Template Selection Logic

```python
if selected_template != "Custom":
    tpl = next((w for w in sample_workloads if w["app_name"] == selected_template), None)
    if tpl:
        reverse_role_map = {v: k for k, v in ROLE_DISPLAY_MAPPING.items()}
        st.session_state.role_display = reverse_role_map.get(
            tpl["role"], list(ROLE_DISPLAY_MAPPING.keys())[0]
        )
```
**Purpose**: 
- `next((w for w in sample_workloads if w["app_name"] == selected_template), None)` - Finds selected template efficiently
- `reverse_role_map = {v: k for k, v in ROLE_DISPLAY_MAPPING.items()}` - Creates reverse mapping for role display
- Auto-populates form fields when user selects a template

#### Form Validation and Processing

```python
if submitted:
    actual_role = ROLE_DISPLAY_MAPPING[role_display]

    st.session_state.workload_config = {
        "role": actual_role,
        "app_name": app_name,
        "gpu_request": gpu_request,
        "memory_request": memory_request,
        "disk_request": disk_request,
        "max_instance_per_node": max_instance_per_node,
        "estimated_execution_hours": estimated_execution_hours,
        "description": description if description else f"{role_display} workload",
    }
    st.session_state.prediction_made = True
    st.rerun()
```
**Purpose**: 
- `actual_role = ROLE_DISPLAY_MAPPING[role_display]` - Converts display name to internal role code
- Stores complete workload configuration in session state
- `st.rerun()` triggers page refresh to show optimization results

#### Results Display and Visualization

```python
# Make prediction
predicted_cpu = optimizer.predict_cpu_request(config)
instance_recommendations = optimizer.optimize_instance_selection([config])
schedule_results = optimizer.schedule_workloads([config])
```
**Purpose**: Orchestrates the complete optimization pipeline: CPU prediction, instance selection, and scheduling optimization.

#### Cost Comparison Visualization

```python
fig_cost = go.Figure(data=[
    go.Bar(
        x=['On-Demand', 'Optimized Spot'],
        y=[job['on_demand_cost'], job['optimized_cost']],
        marker_color=['#f44336', '#4CAF50'],
        text=[f"${job['on_demand_cost']:.2f}", f"${job['optimized_cost']:.2f}"],
        textposition='outside'
    )
])
```
**Purpose**: Creates interactive bar chart comparing on-demand vs optimized spot pricing. Red color indicates higher on-demand cost, green shows optimized savings.

#### Utilization Gauges

```python
fig_cpu = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = instance['cpu_utilization'],
    gauge = {
        'steps': [
            {'range': [0, 50], 'color': "rgba(102, 126, 234, 0.1)"},
            {'range': [50, 80], 'color': "rgba(255, 193, 7, 0.3)"},
            {'range': [80, 100], 'color': "rgba(220, 53, 69, 0.3)"}
        ],
        'threshold': {
            'line': {'color': "#f44336", 'width': 4},
            'value': 90
        }
    }
))
```
**Purpose**: 
- Creates gauge visualization for CPU utilization with color-coded performance zones
- `'steps'` define color ranges: blue (0-50%, good), yellow (50-80%, moderate), red (80-100%, high)
- `'threshold'` at 90% shows red line indicating potential performance risk

#### Scheduling Visualization

```python
# Optimal execution window for task execution is highlighted
start_hour = datetime.strptime(job['scheduled_start'], "%Y-%m-%d %H:%M").hour
end_hour = start_hour + config['estimated_execution_hours']

fig_schedule.add_vrect(
    x0=start_hour, x1=min(end_hour, 24),
    fillcolor="#667eea", opacity=0.2,
    annotation_text="⭐ Optimal Window", 
    annotation_position="top left",
    annotation_font_color="#667eea"
)
```
**Purpose**: 
- `datetime.strptime(job['scheduled_start'], "%Y-%m-%d %H:%M").hour` - Extracts hour from scheduled start time
- `add_vrect()` highlights optimal execution window on 24-hour pricing chart
- `min(end_hour, 24)` prevents highlighting beyond 24-hour boundary

### 4. Instance Pricing Fetcher (`fetch_instance_pricing.py`)

Utility for fetching real-time AWS EC2 pricing data.

#### Region Mapping

```python
def get_region_name(region_code: str) -> str:
    mapping = {
        'us-east-1': 'US East (N. Virginia)',
        'us-east-2': 'US East (Ohio)',
        'us-west-1': 'US West (N. California)',
        # ... more regions
    }
    return mapping.get(region_code, "US East (N. Virginia)")
```
**Purpose**: Maps AWS region codes to pricing API location names. Defaults to N. Virginia for unknown regions.

#### Pricing API Integration

```python
page_iterator = paginator.paginate(
    ServiceCode="AmazonEC2",
    Filters=[
        {"Type": "TERM_MATCH", "Field": "location", "Value": location},
        {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
        {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
        {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
        {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
    ],
    PaginationConfig={"PageSize": 100},
)
```
**Purpose**: 
- Filters AWS pricing data for Linux instances with shared tenancy
- `"preInstalledSw": "NA"` excludes instances with pre-installed software
- `"capacitystatus": "Used"` gets current pricing for actively used capacity
- Pagination handles large result sets efficiently

#### Data Processing and Validation

```python
try:
    vcpu = int(vcpu)
    memory_gb = float(memory.split()[0].replace(",", ""))
except Exception:
    continue

terms = product.get("terms", {}).get("OnDemand", {})
hourly_price = None
for term in terms.values():
    for dim in term.get("priceDimensions", {}).values():
        price = dim.get("pricePerUnit", {}).get("USD")
        if price and price != "0":
            hourly_price = float(price)
            break
```
**Purpose**: 
- `memory.split()[0].replace(",", "")` - Extracts numeric memory value from strings like "32 GiB"
- Searches through pricing terms to find on-demand hourly cost
- `if price and price != "0"` - Ensures valid non-zero pricing

## Workload Templates Structure

```json
{
  "role": "ML_TRAIN",
  "app_name": "tensorflow_training",
  "gpu_request": 4,
  "memory_request": 32000,
  "disk_request": 500000,
  "max_instance_per_node": 1,
  "estimated_execution_hours": 6,
  "description": "Deep learning model training"
}
```
**Purpose**: Defines complete workload specification including resource requirements and execution parameters. Templates provide quick-start configurations for common workload types.

## System Workflow

1. **Prediction Server Startup**: Flask server loads ML models and starts listening
2. **User Configuration**: User selects template or creates custom workload via Streamlit interface  
3. **CPU Prediction**: System calls prediction server with workload parameters
4. **Instance Optimization**: Algorithm finds most cost-effective instance meeting requirements
5. **Schedule Optimization**: System calculates optimal execution timing using spot pricing patterns
6. **Results Display**: Comprehensive dashboard shows predictions, recommendations, and savings

## Key Benefits

- **Cost Optimization**: Achieves 20-70% cost reduction through smart spot pricing
- **Resource Efficiency**: Optimizes CPU and memory utilization for better performance  
- **Intelligent Scheduling**: Finds optimal execution windows to minimize costs
- **User-Friendly Interface**: Streamlit-based dashboard for easy configuration and monitoring
- **Flexible Configuration**: Supports custom workloads and pre-built templates