import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load data
df = pd.read_csv("fraud_data.csv")

# Load model
model = joblib.load("model.pkl")

# Title
st.title("🔍 Transaction Explorer")

# Sidebar filters
st.sidebar.header("Filters")

min_amount = st.sidebar.slider(
    "Minimum Amount",
    0,
    int(df["TransactionAmt"].max()),
    0
)

max_amount = st.sidebar.slider(
    "Maximum Amount",
    0,
    int(df["TransactionAmt"].max()),
    int(df["TransactionAmt"].max())
)

# Filter transactions
filtered_df = df[
    (df["TransactionAmt"] >= min_amount) &
    (df["TransactionAmt"] <= max_amount)
]

# Show filtered table
st.subheader("Filtered Transactions")
st.dataframe(filtered_df)

# Search transaction
st.subheader("Search Transaction")

transaction_id = st.text_input(
    "Enter TransactionID"
)

if transaction_id:

    try:
        transaction_id = int(transaction_id)

        transaction = df[
            df["TransactionID"] == transaction_id
        ]

        if not transaction.empty:

            st.write(transaction)

            # Prepare input for model
            X = transaction.drop(
                columns=["TransactionID", "isFraud"]
            )

            # Predict fraud probability
            probability = model.predict_proba(X)[0][1]

            st.subheader("Fraud Risk Score")

            st.progress(float(probability))

            st.write(
                f"Fraud Probability: {probability:.2f}"
            )

            # Risk level
            if probability >= 0.75:
                st.error("🔴 High Risk")

            elif probability >= 0.40:
                st.warning("🟡 Medium Risk")

            else:
                st.success("🟢 Low Risk")

        else:
            st.error("Transaction not found")

    except:
        st.error("Invalid TransactionID")

# Scatter plot
st.subheader("Transaction Analysis")

if "HourOfDay" in df.columns:

    fig = px.scatter(
        filtered_df,
        x="HourOfDay",
        y="TransactionAmt",
        color="isFraud",
        title="Transaction Amount vs Hour Of Day"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )