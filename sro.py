import pickle
import pandas as pd
import json
import requests
from datetime import datetime, timedelta
from enum import Enum
import math
import csv
import os
from typing import Dict, List, Optional, Tuple

class NodeRole(str, Enum):
    HN = "HN"
    CN = "CN"

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

class SmartResourceOptimizer:
    def __init__(self, pred_endpoint=None, timeout=10, csv_file="./data/csv/aws_pricing.csv"):
        if pred_endpoint is None:
            pred_endpoint = os.getenv("PREDICTOR_URL", "http://127.0.0.1:5000/predict")
        
        self.pred_endpoint = pred_endpoint
        self.timeout = timeout
        self.csv_file = csv_file
        self.instance_pricing = self._load_csv_pricing()

    def _load_csv_pricing(self) -> Dict:
        try:
            if not os.path.exists(self.csv_file):
                print(f"Pricing csv file {self.csv_file} not found.")
                return
            
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
            
            print(f"Loaded {len(pricing_data)} pricing entries from CSV: {self.csv_file}")
            return pricing_data
            
        except Exception as e:
            print(f"Failed to load CSV pricing data: {e}")
            return

    def get_all_instances(self) -> List[Dict]:
        instances = []
        try:
            with open(self.csv_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    instances.append({
                        'instance_name': row['instance_name'],
                        'cpu_count': int(row['cpu_count']),
                        'memory_gb': int(row['memory_gb']),
                        'hourly_cost': float(row['hourly_cost'])
                    })
        except Exception as e:
            print(f"Error reading instances: {e}")
        return instances

    def map_role(self, workload_config):
        original_role = workload_config.get("role", "unknown")
        mapped_role = ROLE_MAPPING.get(original_role, NodeRole.CN)
        return mapped_role.value

    def predict_cpu_request(self, workload_config):
        try:
            mapped_role = self.map_role(workload_config)

            payload = {
                "gpu_request": workload_config.get("gpu_request", 0),
                "disk_request": workload_config.get("disk_request", 0), 
                "memory_request": workload_config.get("memory_request", 0),
                "max_instance_per_node": workload_config.get("max_instance_per_node", 1),
                "role": mapped_role,
                "app_name": workload_config.get("app_name", "unknown")
            }
            
            for key, value in payload.items():
                if pd.isna(value) or value is None:
                    if key in ["gpu_request", "disk_request", "memory_request", "max_instance_per_node"]:
                        payload[key] = 0
                    else:
                        payload[key] = "UNKNOWN"
            
            payload["role"] = str(payload["role"])
            payload["app_name"] = str(payload["app_name"])
            
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
            else:
                print(f"API call failed with status code {response.status_code}: {response.text}")
                return 1
                
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return 1
        except Exception as e:
            print(f"Unexpected error during API call: {e}")
            return 1
    
    def optimize_instance_selection(self, workload_configs):
        results = []
        all_instances = self.get_all_instances()
        
        for i, config in enumerate(workload_configs):
            predicted_cpu = self.predict_cpu_request(config)
            memory_needed = config["memory_request"]
            
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
            
            if best_instance:
                execution_hours = config.get("estimated_execution_hours", 8)
                total_cost = min_cost * execution_hours
                
                results.append({
                    'workload_id': f"job_{i+1}",
                    'predicted_cpu': predicted_cpu,
                    'memory_needed_mb': memory_needed,
                    'recommended_instance': best_instance,
                    'estimated_cost': total_cost,
                    'config': config
                })
            else:
                # if no suitable instance is found
                results.append({
                    'workload_id': f"job_{i+1}",
                    'predicted_cpu': predicted_cpu,
                    'memory_needed_mb': memory_needed,
                    'recommended_instance': None,
                    'estimated_cost': None,
                    'config': config,
                    'error': f"No instance available for {predicted_cpu} CPUs and {memory_needed} MB memory"
                })
        
        return results
    
    def schedule_workloads(self, workload_configs, time_horizon_hours=24):
        base_time = datetime.now()
        all_instances = self.get_all_instances()
        
        # Hourly cost multipliers simulating on demand instance pricing patterns
        hourly_multipliers = [
            0.3, 0.25, 0.2, 0.2, 0.25, 0.4,
            0.6, 0.8, 1.0, 1.0, 0.9, 0.8,
            0.9, 1.0, 1.0, 0.9, 0.8, 0.7,
            0.6, 0.5, 0.4, 0.35, 0.3, 0.3
        ]
        
        scheduled_jobs = []
        total_savings = 0
        
        for i, config in enumerate(workload_configs):
            predicted_cpu = self.predict_cpu_request(config)
            execution_duration = config.get("estimated_execution_hours", 8)
            
            suitable_instance = None
            # Find the cheapest suitable instance
            min_base_cost = float('inf')
            for instance in all_instances:
                if (instance['cpu_count'] >= predicted_cpu and 
                    (instance['memory_gb'] * 1024) >= config["memory_request"] and
                    instance['hourly_cost'] < min_base_cost):
                    suitable_instance = instance
                    min_base_cost = instance['hourly_cost']
            
            if not suitable_instance:
                scheduled_jobs.append({
                    'job_id': f"job_{i+1}",
                    'predicted_cpu': predicted_cpu,
                    'instance_name': 'N/A',
                    'scheduled_start': 'N/A',
                    'scheduled_end': 'N/A',
                    'optimized_cost': None,
                    'on_demand_cost': None,
                    'savings': 0,
                    'config': config,
                    'error': f"No suitable instance for {predicted_cpu} CPUs and {config['memory_request']} MB memory"
                })
                continue
            
            base_cost = suitable_instance['hourly_cost']
            instance_name = suitable_instance['instance_name']
            
            best_start_hour = 0
            min_total_cost = float('inf')
            
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
            
            on_demand_cost = base_cost * execution_duration
            savings = on_demand_cost - min_total_cost
            total_savings += savings
            
            start_time = base_time + timedelta(hours=best_start_hour)
            end_time = start_time + timedelta(hours=execution_duration)
            
            scheduled_jobs.append({
                'job_id': f"job_{i+1}",
                'predicted_cpu': predicted_cpu,
                'instance_name': instance_name,
                'scheduled_start': start_time.strftime("%Y-%m-%d %H:%M"),
                'scheduled_end': end_time.strftime("%Y-%m-%d %H:%M"),
                'optimized_cost': min_total_cost,
                'on_demand_cost': on_demand_cost,
                'savings': savings,
                'config': config
            })
        
        return {
            'scheduled_jobs': scheduled_jobs,
            'total_savings': total_savings,
            'optimization_summary': {
                'total_jobs': len(workload_configs),
                'total_savings_usd': round(total_savings, 2),
                'average_savings_per_job': round(total_savings / len(workload_configs), 2) if workload_configs else 0
            }
        }

def run():
    optimizer = SmartResourceOptimizer(csv_file="./data/csv/aws_pricing.csv")
    
    sample_workloads = [
        {
            "role": "ML_TRAIN",
            "app_name": "tensorflow_training",
            "gpu_request": 4,
            "memory_request": 32000,
            "disk_request": 500000,
            "max_instance_per_node": 1,
            "estimated_execution_hours": 6,
            "description": "Deep learning model training"
        },
        {
            "role": "ML_INFERENCE",
            "app_name": "model_serving",
            "gpu_request": 1,
            "memory_request": 16000,
            "disk_request": 100000,
            "max_instance_per_node": 2,
            "estimated_execution_hours": 4,
            "description": "Real-time model inference"
        },
        {
            "role": "ETL",
            "app_name": "data_preprocessing", 
            "gpu_request": 0,
            "memory_request": 8000,
            "disk_request": 200000,
            "max_instance_per_node": 4,
            "estimated_execution_hours": 3,
            "description": "ETL data transformation"
        },
        {
            "role": "ML_TRAIN",
            "app_name": "hyperparameter_tuning",
            "gpu_request": 2,
            "memory_request": 24000,
            "disk_request": 300000,
            "max_instance_per_node": 1,
            "estimated_execution_hours": 12,
            "description": "ML hyperparameter optimization"
        },
        {
            "role": "ETL",
            "app_name": "batch_processing",
            "gpu_request": 0,
            "memory_request": 12000,
            "disk_request": 800000,
            "max_instance_per_node": 3,
            "estimated_execution_hours": 5,
            "description": "Large-scale batch data processing"
        }
    ]
    
    print("Smart Resource Optimization System")
    print("=" * 50)
    
    print("\nCPU Resource Predictions:")
    for i, config in enumerate(sample_workloads):
        predicted_cpu = optimizer.predict_cpu_request(config)
        print(f"Job {i+1} ({config['description']} - {config['role']}): {predicted_cpu} cores")
    
    print("\nOptimal Instance Recommendations:")
    instance_recommendations = optimizer.optimize_instance_selection(sample_workloads)
    
    total_cost = 0
    for rec in instance_recommendations:
        if rec.get('error'):
            print(f"{rec['workload_id']}: {rec['error']}")
            continue
            
        inst = rec['recommended_instance']
        print(f"{rec['workload_id']} ({rec['config']['description']}): {inst['instance_name']} ({inst['cpu_count']} vCPU, {inst['memory_gb']} GB RAM), ${rec['estimated_cost']:.2f}")
        total_cost += rec['estimated_cost']
    
    print(f"\nTotal On-Demand Cost: ${total_cost:.2f}")
    
    print("\nScheduling Optimization with Spot Pricing:")
    schedule_results = optimizer.schedule_workloads(sample_workloads)
    
    for job in schedule_results['scheduled_jobs']:
        if job.get('error'):
            continue
        print(f"{job['job_id']} ({job['config']['description']}): {job['instance_name']} - Start {job['scheduled_start']}, Save ${job['savings']:.2f}")
    
    summary = schedule_results['optimization_summary']
    print(f"\nOptimization Summary:")
    print(f"Total Savings: ${summary['total_savings_usd']}")
    savings_pct = (summary['total_savings_usd'] / total_cost) * 100 if total_cost > 0 else 0
    print(f"Cost Reduction: {savings_pct:.1f}%")
    print(f"Average Savings per Job: ${summary['average_savings_per_job']}")

if __name__ == "__main__":
    run()