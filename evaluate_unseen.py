import pandas as pd 
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

# 1. Load separate training and testing datasets
train_df = pd.read_csv("transactions.csv")
test_df = pd.read_csv("test_transactions.csv")

# 2. Select the same features for both datasets
features = ["amount", "hour"]

X_train = train_df[features]
X_test = test_df[features]

# 3. Train ONLY on the training data
model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

model.fit(X_train)

# 4. Predict on previously unseen transactions
test_df["prediction"] = model.predict(X_test)

# Convert -1 to 1 for flagged transactions
test_df["flagged"] = (
    test_df["prediction"] == -1
).astype(int)

# 5. Compare predictions against out synthetic labels
actual = test_df["injected_anomaly"]
predicted = test_df["flagged"]

precision = precision_score(
    actual, predicted, zero_division=0
)

recall = recall_score(
    actual, predicted, zero_division=0
)

false_positives = (
    (actual == 0) & (predicted == 1)
).sum

# 6. Display our results 
print("\n=== UNSEEN DATA EVALUATION ===")
print(f"Test transactions: {len(test_df)}")
print(f"Flagged transactions: {predicted.sum()}")
print(f"Precision: {precision:.1%}")
print(f"Recall: {recall:.1%}")
print(f"False positives: {false_positives}")

print("\nResults by scenario:")
print(
    test_df.groupby("scenario")["flagged"]
    .agg(["sum", "count"])
)

# 7. Save results for future dashboard improvements
test_df.to_csv("unseen_test_results.csv", index=False)


