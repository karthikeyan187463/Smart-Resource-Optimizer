import boto3
import json
import csv
from datetime import datetime

def get_region_name(region_code: str) -> str:
    mapping = {
        'us-east-1': 'US East (N. Virginia)',
        'us-east-2': 'US East (Ohio)',
        'us-west-1': 'US West (N. California)',
        'us-west-2': 'US West (Oregon)',
        'eu-west-1': 'Europe (Ireland)',
        'eu-west-2': 'Europe (London)',
        'eu-west-3': 'Europe (Paris)',
        'eu-central-1': 'Europe (Frankfurt)',
        'ap-southeast-1': 'Asia Pacific (Singapore)',
        'ap-southeast-2': 'Asia Pacific (Sydney)',
        'ap-northeast-1': 'Asia Pacific (Tokyo)',
        'ap-south-1': 'Asia Pacific (Mumbai)',
        'sa-east-1': 'South America (Sao Paulo)',
        'ca-central-1': 'Canada (Central)',
    }
    return mapping.get(region_code, "US East (N. Virginia)")

def fetch_pricing(region: str):
    location = get_region_name(region)

    client = boto3.client("pricing", region_name="us-east-1")

    paginator = client.get_paginator("get_products")
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

    results = []
    seen = set()

    for page in page_iterator:
        for price_item in page["PriceList"]:
            product = json.loads(price_item)
            attrs = product.get("product", {}).get("attributes", {})
            instance_type = attrs.get("instanceType")
            vcpu = attrs.get("vcpu")
            memory = attrs.get("memory")

            if not instance_type or instance_type in seen:
                continue
            seen.add(instance_type)

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
                if hourly_price:
                    break

            if hourly_price:
                results.append(
                    {
                        "instance_name": instance_type,
                        "cpu_count": vcpu,
                        "memory_gb": int(memory_gb),
                        "hourly_cost": hourly_price,
                    }
                )

    return sorted(results, key=lambda x: (x["cpu_count"], x["hourly_cost"]))

def save_to_csv(data, output_file: str):
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["instance_name", "cpu_count", "memory_gb", "hourly_cost"]
        )
        writer.writeheader()
        writer.writerows(data)

def main():
    region = "us-east-1"
    output_file = "data/csv/aws_pricing.csv"

    print(f"Fetching pricing for region {region}...")
    data = fetch_pricing(region)

    print(f"Saving {len(data)} records to {output_file}")
    save_to_csv(data, output_file)

    print("Done!")

if __name__ == "__main__":
    main()
