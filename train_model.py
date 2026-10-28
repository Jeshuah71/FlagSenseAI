import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Load our transaction dataset
df = pd.read_csv("transactions.csv")

print("Dataset preview: ")
print(df.head())

# 2. Select the features for the model
X = df[["amount", "hour"]]

# 3. Create the Isolation Forest model
model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
    
)

# Train the model
model.fit(X)

# 5. Predict which of the transactions are unusual
df["prediction"] = model.predict(X)

# 6. Calculate anomaly scores 
df["anomaly_score"] = model.decision_function(X)

# 7. Display flagged transactions
flagged = df[df["prediction"] == -1]
flagged = flagged.sort_values("anomaly_score")

print("\nFlagged transactions:")
print(flagged[["amount", "hour", "anomaly_score"]])

# Save the results 
df.to_csv("analized_transactions.csv", index=False)

print(f"\nTotal transactions: {len(df)}")
print(f"Flagged transactions: {len(flagged)}")


