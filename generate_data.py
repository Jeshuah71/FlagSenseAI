import numpy as np
import pandas as pd

# Create a random number generator
rng = np.random.default_rng(42)

# Generate 500 typical transactions 
normal = pd.DataFrame ({
    "amount": rng.normal(45, 18, 500).clip(5).round(2),
    "hour": rng.integers(8, 22, 500),
    "injected_anomaly": 0
})

# Generate 10 deliberately unusual transactions
unusual = pd.DataFrame({
    "amount": rng.uniform(500, 1800, 10).round(2),
    "hour": rng.integers(0, 5, 10),
    "injected_anomaly": 1
})

# Combine and shuffle the transactions
transactions = pd.concat(
    [normal, unusual], ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

# Save as CSV
transactions.to_csv("transactions.csv", index=False)

print(transactions.head())
print(f"\nCreated {len(transactions)} transactions.")

