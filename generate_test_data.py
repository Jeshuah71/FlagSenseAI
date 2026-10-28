import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)

# 300 typical transactions
normal = pd.DataFrame({
    "amount": rng.normal(45, 18, 300).clip(5).round(2),
    "hour": rng.integers(8, 22, 300),
    "injected_anomaly": 0,
    "scenario": "Typical"
})

# 10 obvious unusual transactions
obvious = pd.DataFrame({
    "amount": rng.uniform(500, 1800, 10).round(2),
    "hour": rng.integers(0, 5, 10),
    "injected_anomaly": 1,
    "scenario": "Obvious anomaly"
})

# 10 more challenging unusual transactions
subtle = pd.DataFrame({
    "amount": rng.uniform(75, 125, 10).round(2),
    "hour": rng.integers(8, 22, 10),
    "injected_anomaly": 1,
    "scenario": "Subtle anomaly"
})

# Combine and shuffle the test data
test_df = pd.concat(
    [normal, obvious, subtle],
    ignore_index=True  
).sample(frac=1, random_state=17).reset_index(drop=True)

test_df.to_csv("test_transactions.csv", index=False)

print(test_df["scenario"].value_counts())
print(f"\nCreated {len(test_df)} new test transactions.")