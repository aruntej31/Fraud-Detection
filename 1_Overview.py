import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv("fraud_data.csv")

# Metrics
total_transactions = len(df)

fraud_count = df["isFraud"].sum()

detection_rate = (fraud_count / total_transactions) * 100

avg_fraud_amt = df[df["isFraud"] == 1]["TransactionAmt"].mean()

# Title
st.title("Fraud Detection Overview")

# Metric Cards
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Fraud Count",
    f"{fraud_count:,}"
)

col3.metric(
    "Detection Rate",
    f"{detection_rate:.2f}%"
)

col4.metric(
    "Average Fraud Amount",
    f"${avg_fraud_amt:.2f}"
)

# Fraud Distribution Chart
st.subheader("Fraud vs Non-Fraud Distribution")

fraud_dist = df["isFraud"].value_counts().reset_index()

fraud_dist.columns = ["Fraud", "Count"]

fig = px.pie(
    fraud_dist,
    names="Fraud",
    values="Count",
    hole=0.4,
    title="Fraud Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# Transaction Amount Histogram
st.subheader("Transaction Amount Distribution")

fig2 = px.histogram(
    df,
    x="TransactionAmt",
    color="isFraud",
    nbins=50,
    marginal="box",
    title="Fraud vs Non-Fraud Transaction Amount"
)

st.plotly_chart(fig2, use_container_width=True)