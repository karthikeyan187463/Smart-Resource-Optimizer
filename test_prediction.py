import json
import requests

data_path = "./data/json/data.json"
pred_endpoint = "http://127.0.0.1:5000/predict"

with open(data_path, "r") as f:
    payload = json.load(f)

response = requests.post(pred_endpoint, json=payload, timeout=10)

print("Status Code:", response.status_code)
try:
    print("Response:", response.json())
except json.JSONDecodeError:
    print("Response not JSON:", response.text)
