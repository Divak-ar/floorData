import requests
import os
from pathlib import Path
import json

# ✅ Path to JSON file or directly get response from API
API_URL = "https://datasets-server.huggingface.co/rows"
params = {
    "dataset": "zimhe/pseudo-floor-plan-12k",
    "config": "default",
    "split": "train",
    "offset": 10,
    "length": 100  # Adjust as needed
}

# ✅ Output folder
download_dir = Path("huggingface_dataset")
(download_dir / "plans").mkdir(parents=True, exist_ok=True)
(download_dir / "walls").mkdir(parents=True, exist_ok=True)

# ✅ Get the response
response = requests.get(API_URL, params=params)
data = response.json()

# print(data)
print(json.dumps(data, indent=2))


# ✅ Loop and download
for i, row in enumerate(data['rows']):
    idx = row['row']['indices']
    
    plan_url = row['row']['plans']['src']
    wall_url = row['row']['walls']['src']
    
    # Download plan image
    plan_resp = requests.get(plan_url)
    with open(download_dir / "plans" / f"{idx}.png", "wb") as f:
        f.write(plan_resp.content)
    
    # Download wall mask image
    wall_resp = requests.get(wall_url)
    with open(download_dir / "walls" / f"{idx}.jpg", "wb") as f:
        f.write(wall_resp.content)

    print(f"✅ Downloaded: {idx}")
