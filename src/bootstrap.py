import os
import sys

def run_bootstrap():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    os.chdir(root)

    print("[Bootstrap] Generating synthetic training data...")
    from scripts.generate_data import generate_dataset
    import json
    import pandas as pd

    OFFICERS = [
        {"id": "OFF001", "name": "Rajesh Kumar",    "domain": "infrastructure"},
        {"id": "OFF002", "name": "Priya Sharma",     "domain": "finance"},
        {"id": "OFF003", "name": "Ankit Verma",      "domain": "sanitation"},
        {"id": "OFF004", "name": "Sunita Patel",     "domain": "electricity"},
        {"id": "OFF005", "name": "Mohan Das",        "domain": "water"},
        {"id": "OFF006", "name": "Kavitha Rao",      "domain": "roads"},
        {"id": "OFF007", "name": "Deepak Nair",      "domain": "noise_pollution"},
        {"id": "OFF008", "name": "Fatima Sheikh",    "domain": "corruption"},
    ]

    os.makedirs("data", exist_ok=True)
    df = generate_dataset(800)
    df.to_csv("data/complaints.csv", index=False)
    with open("data/officers.json", "w") as f:
        json.dump(OFFICERS, f, indent=2)

    print("[Bootstrap] Training models...")
    from scripts.train import main as train_main
    train_main()
    print("[Bootstrap] Done.")
