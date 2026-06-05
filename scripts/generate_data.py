import pandas as pd
import random
import json
import os

random.seed(42)

OFFICERS = [
    {"id": "OFF001", "name": "Rajesh Kumar",     "domain": "infrastructure"},
    {"id": "OFF002", "name": "Priya Sharma",      "domain": "finance"},
    {"id": "OFF003", "name": "Ankit Verma",       "domain": "sanitation"},
    {"id": "OFF004", "name": "Sunita Patel",      "domain": "electricity"},
    {"id": "OFF005", "name": "Mohan Das",         "domain": "water"},
    {"id": "OFF006", "name": "Kavitha Rao",       "domain": "roads"},
    {"id": "OFF007", "name": "Deepak Nair",       "domain": "noise_pollution"},
    {"id": "OFF008", "name": "Fatima Sheikh",     "domain": "corruption"},
]

# (text_template, officer_domain, priority, eta_days)
COMPLAINT_TEMPLATES = [
    # Infrastructure
    ("The road in front of my house has a large pothole causing accidents", "roads", "High", 3),
    ("The bridge near sector 5 has cracks and is dangerous", "infrastructure", "High", 2),
    ("Street lights have been non-functional for 3 weeks", "electricity", "Medium", 5),
    ("The footpath near market is completely broken", "roads", "Medium", 7),
    ("Road construction debris left blocking traffic for days", "roads", "Low", 10),
    # Electricity
    ("Power outage in our area for the last 12 hours, no response from helpline", "electricity", "High", 1),
    ("Transformer at main junction is sparking dangerously", "electricity", "High", 1),
    ("Electricity bill has wrong readings for three consecutive months", "finance", "Medium", 7),
    ("Wires hanging loose near the park, safety hazard", "electricity", "High", 2),
    ("Meter box outside our colony is damaged and exposed", "electricity", "Medium", 5),
    # Water
    ("No water supply since 2 days in entire block", "water", "High", 2),
    ("Sewage water mixing with drinking water supply", "water", "High", 1),
    ("Water pressure is very low, barely usable", "water", "Medium", 5),
    ("Water supply comes only for 30 minutes a day", "water", "Medium", 7),
    ("Pipeline leak on main road wasting water for weeks", "water", "Medium", 4),
    # Sanitation
    ("Garbage not collected for 10 days, health hazard", "sanitation", "High", 2),
    ("Dead animal carcass near playground not removed", "sanitation", "High", 1),
    ("Drainage overflow on main road due to blocked drains", "sanitation", "High", 3),
    ("Public toilet near bus stand is non-functional and filthy", "sanitation", "Medium", 5),
    ("Stray dogs attacking people near the school", "sanitation", "High", 2),
    # Finance/Corruption
    ("Bribe demanded for property registration document", "corruption", "High", 3),
    ("Government scheme benefits not received despite eligibility", "finance", "Medium", 14),
    ("Double taxation applied without proper reason", "finance", "Medium", 10),
    ("Contractor taking money but not completing work", "corruption", "High", 5),
    ("Ration shop owner selling goods at higher than official price", "corruption", "High", 3),
    # Noise/Environment
    ("Construction noise after 10pm violating norms daily", "noise_pollution", "Medium", 7),
    ("Factory releasing smoke causing breathing problems", "noise_pollution", "High", 3),
    ("Loud music from nearby wedding hall till 3am every night", "noise_pollution", "Medium", 5),
    ("Illegal dumping of chemical waste near river", "noise_pollution", "High", 2),
    ("Open burning of garbage causing pollution", "noise_pollution", "Medium", 5),
    # Multilingual samples (Hindi/Odia transliterated)
    ("Mere ghar ke paas ki sadak mein bahut bada gadha hai", "roads", "High", 3),
    ("Bijli 2 din se nahi hai, koi sunne wala nahi", "electricity", "High", 1),
    ("Pani supply bilkul band ho gayi hai ward mein", "water", "High", 2),
    ("Kachra uthane wale 1 hafte se nahi aaye", "sanitation", "High", 2),
    ("Amare ghara saamne rasta beshi kharapa", "roads", "Medium", 7),
]

AUGMENTATIONS = [
    "Please look into this urgently.",
    "This is affecting many residents.",
    "Multiple complaints filed but no action taken.",
    "Children and elderly people are at risk.",
    "Kindly resolve at earliest.",
    "This has been going on for too long.",
    "Local authorities have not responded.",
    "We are suffering due to this issue.",
]

def generate_dataset(n=800):
    rows = []
    for i in range(n):
        tmpl = random.choice(COMPLAINT_TEMPLATES)
        text, domain, priority, eta = tmpl
        # Augment text slightly
        if random.random() > 0.4:
            text = text + " " + random.choice(AUGMENTATIONS)
        # Add some noise to ETA (±1-2 days)
        eta_noisy = max(1, eta + random.randint(-1, 2))
        # Map domain -> officer
        officer = next((o for o in OFFICERS if o["domain"] == domain), random.choice(OFFICERS))
        rows.append({
            "complaint_id": f"CMP{1000+i}",
            "text": text,
            "officer_id": officer["id"],
            "officer_name": officer["name"],
            "domain": domain,
            "priority": priority,
            "eta_days": eta_noisy,
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(800)
    df.to_csv("data/complaints.csv", index=False)
    # Save officer registry
    with open("data/officers.json", "w") as f:
        json.dump(OFFICERS, f, indent=2)
    print(f"Generated {len(df)} complaints → data/complaints.csv")
    print(f"Priority distribution:\n{df['priority'].value_counts()}")
    print(f"Officer distribution:\n{df['officer_id'].value_counts()}")
