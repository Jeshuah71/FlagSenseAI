import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest

# 1. Configure the application
st.set_page_config(
    page_title="FlagSense AI",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("FlagSense AI")
st.write("Explore unusual financial transactions using Machine Learning.")

# 2 Allow users to upload their own dataset 
st.sidebar.subheader("Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload Transactions",
    type=["csv"],
    help="CSV must contain amount and hour columns."
)

# Use our sample dataset if no file is uploaded
try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.caption(f"Data source: {uploaded_file.name}")
    else:
        df = pd.read_csv("transactions.csv")
        st.caption("Data source: Synthetic sample transactions")
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
    st.error("Could not read this CSV file. Please check its format.")
    st.stop()
    
# Validate required columns
required_columns = {"amount", "hour"}

if not required_columns.issubset(df.columns):
    st.error("Your CSV must contain 'amount' and 'hour' columns.")
    st.stop()
    
# Convert features to numbers
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["hour"] = pd.to_numeric(df["hour"], errors="coerce")

# Remove invalid transactions
invalid_rows = (
    df["amount"].isna() | df["hour"].isna() | (df["amount"] < 0) | ~df["hour"].between(0, 23) | (df["hour"] % 1 != 0)
)

removed = int(invalid_rows.sum())
df = df.loc[~invalid_rows].copy()

if removed > 0:
    st.warning(f"Excluded {removed} rows with invalid data.")
    
if len(df) < 20:
    st.error("Please provide at least 20 valid transactions.")
    st.stop()
    
# Only use evaluation labels if they are valid
if "injected_anomaly" is df.columns:
    labels = pd.to_numeric(df["injected_anomaly"], errors="coerce")
    
    if labels.isin([0, 1]).all():
        df["injected_anomaly"] = labels.astype(int)
        
    else:
        df = df.drop(columns=["injected_anomaly"])
        st.warning("Invalid evaluation labels were ignored.")


# 3. Let the user control sensitivity
contamination = st.sidebar.slider(
    "Anomaly Detection Sensitivity",
    min_value=0.01,
    max_value=0.10,
    value=0.02,
    step=0.01
)

# 4. Train the model 
X = df[["amount", "hour"]]

model = IsolationForest(
    n_estimators=100,
    contamination=contamination,
    random_state=42
)

df["prediction"] = model.fit_predict(X)
df["anomaly_score"] = model.decision_function(X)

# 5. Convert numerical predictions into readable lables
df["status"] = df["prediction"].map({
    1:"Typical",
    -1: "Flagged"
})

# 6. Display summary statistics
total = len(df)
flagged_count = (df["prediction"] == -1).sum()
flagged_rate = flagged_count / total * 100

col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", total)
col2.metric("Flagged Transactions", flagged_count)
col3.metric("Flagged Rate", f"{flagged_rate:.1f}%")

# 7. Visulize the transactions
st.subheader("Transaction Anomaly Analysis")

fig = px.scatter(
    df, 
    x="hour",
    y="amount",
    color="status",
    hover_data=["anomaly_score"],
    color_discrete_map={
        "Typical": "#119e75",
        "Flagged": "#f97316"
    },
    
    labels={
        "hour": "Hour of Day",
        "amount": "Transactions Amount ($)",
        "status": "Classification"
    }
)


# Improve chart appearance on desktop and mobile
fig.update_layout(
    height=370,
    autosize=True,
    margin=dict(l=5, r=5, t=45, b=5),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
    font=dict(size=12)
)

fig.update_traces(
    marker=dict(size=7, opacity=0.8)
)


st.plotly_chart(fig, use_container_width=True)

# 8. Show flagged transactions
st.subheader("Transactions for Review")

flagged = df[df["prediction"] == -1]
flagged = flagged.sort_values("anomaly_score")

st.dataframe(
    flagged[["amount", "hour", "anomaly_score"]],
    hide_index=True,
    height=300,
    use_container_width=True,
    column_config={
        "amount": st.column_config.NumberColumn(
            "Amount",
            format="$%.2f"
        ),
        "hour": st.column_config.NumberColumn(
            "Hour"
        ),
        "anomaly_score": st.column_config.NumberColumn(
            "Anomaly Score",
            format="%.4f"
        )
    }
)

st.caption(
    "Flagged transactions are statistical anomalies, "
    "not confirmed fraudulent activity."
)

# 9. Export analyzed transactions
st.subheader("Export Analysis")

csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Analysis as CSV",
    data=csv_data,
    file_name="flagsense_analysis.csv",
    mime="text/csv",
    on_click="ignore"
)

# Evaluate predictions using our synthetic lables
if "injected_anomaly" in df.columns:
    st.subheader("Training Data Evaluation")
    
    # Actual answers from the dataset
    actual = df["injected_anomaly"]
    
    # Convert model predictions: -1 becomes 1 (flagged)
    predicted = (df["prediction"] == -1).astype(int)
    
    # Count correct and incorrect predictions
    true_positives = ((actual == 1) & (predicted == 1)).sum()
    false_positives = ((actual == 0) & (predicted == 1)).sum()
    false_negatives = ((actual == 1) & (predicted == 0)).sum()
    
    # Calculate evaluation metrics
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Precision", f"{precision:.1%}")
    col2.metric("Recall", f"{recall:.1%}")
    col3.metric("False Positives", false_positives)
    
    st.caption(
        "Evaluation uses known labels from simulated data. "
        "Results do not establish real-world fraud detection accuracy."
    )
    
    
    # Explain evaluation metrics
    with st.expander("How to interpret these results"):
        st.write(
            f"True Positives: {true_positives} "
            "injected anomalies correctly detected."
        )

        st.write(
            f"False Positives: {false_positives} "
            "typical transactions incorrectly flagged."
        )

        st.write(
            f"False Negatives: {false_negatives} "
            "injected anomalies missed by the model."
        )

        st.markdown(
            "**Precision:** Of the transactions flagged, "
            "how many were actually injected anomalies?"
        )

        st.markdown(
            "**Recall:** Of all injected anomalies, "
            "how many did the model detect?"
        )


# 10 Evaluate model on previously unseen transactions
if uploaded_file is None:
    st.subheader("Unseen Data Evaluation")
    
    try:
        test_df = pd.read_csv("test_transactions.csv")
        
    except FileNotFoundError:
        st.info(
            "Run python generate_test_data.py to create "
            "the unseen testing dataset"
        )
        
    else:
        # Use the same features as training
        X_test = test_df[["amount", "hour"]]
        
        # Predict WITHOUT training the model again
        test_df["flagged"] = (
            model.predict(X_test) == -1
        ).astype(int)
        
        # Compare with the known synthetic labels
        actual = test_df["injected_anomaly"]
        predicted = test_df["flagged"]
        
        true_positives = int(
            ((actual == 1) & (predicted == 1)).sum()
        )
        
        false_positives = int(
            ((actual == 0) & (predicted == 1)).sum()
        )
        
        false_negatives = int(
            ((actual == 1) & (predicted ==0)).sum()
        )
        
        # Calculate precision and recall 
        precision = true_positives / max(
            true_positives + false_positives, 1
        )

        recall = true_positives / max(
            true_positives + false_negatives, 1
        )

        # Display evaluation metrics
        col1, col2, col3 = st.columns(3)

        col1.metric("Test Precision", f"{precision:.1%}")
        col2.metric("Test Recall", f"{recall:.1%}")
        col3.metric("Test False Positives", false_positives)

        # Compare results by transaction scenario
        st.markdown("#### Detection by Scenario")

        scenario_results = (
            test_df.groupby("scenario")["flagged"]
            .agg(["sum", "count"])
            .reset_index()
        )

        scenario_results.columns = [
            "Scenario", "Flagged", "Total"
        ]

        scenario_results["Flagged Rate (%)"] = (
            scenario_results["Flagged"]
            / scenario_results["Total"] * 100
        ).round(1)

        # Combine counts and percentages into one column
        scenario_results["Result"] = scenario_results.apply(
            lambda row: (
                f'{int(row["Flagged"])} / {int(row["Total"])}'
                f'({row["Detection Rate (%)"]:.1f}%)'
            ),
            axis=1
        )
        
        st.dataframe(
            scenario_results[["Scenario", "Result"]],
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "This evaluation uses new synthetic transactions "
            "that were not used to train the model. "
            "Results are not real-world fraud detection accuracy."
        )


# Explain how FlagSense AI should be interpreted
st.divider()
st.subheader("About FlagSense AI")

with st.expander("How does the model work?"):
    st.write(
        "FlagSense AI uses Isolation Forest, an unsupervised "
        "machine learning algorithm that identifies unusual "
        "patterns in transaction data."
    )

    st.write(
        "The current model uses transaction amount and "
        "hour of day as features."
    )

with st.expander("Model limitations"):
    st.markdown("""
    - Flagged transactions are not necessarily fraudulent.
    - The model only analyzes amount and transaction time.
    - Precision and recall require trustworthy known labels.
    - Our demonstration data is synthetic.
    - Real-world performance has not been established.
    - Some unusual activity may look statistically normal.
    """)

st.caption(
    "FlagSense AI is an educational machine learning project, "
    "not a production fraud detection system."
)
