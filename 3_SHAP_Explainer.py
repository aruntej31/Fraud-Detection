import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("fraud_data.csv")

# Load Model
model = joblib.load("model.pkl")

# Load SHAP Explainer
explainer = joblib.load("shap_explainer.pkl")

# Title
st.title("SHAP Fraud Explanation")

# User Input
transaction_id = st.text_input(
    "Enter TransactionID"
)

if transaction_id:

    try:

        transaction_id = int(transaction_id)

        row = df[
            df["TransactionID"] == transaction_id
        ]

        if len(row) > 0:

            st.subheader("Transaction Details")

            st.write(row)

            # Prepare features
            X = row.drop(columns=["isFraud"])

            # Fraud Probability
            probability = model.predict_proba(X)[:,1][0]

            st.subheader("Fraud Probability")

            st.write(f"{probability:.2f}")

            # =========================
            # SHAP VALUES
            # =========================

            shap_values = explainer.shap_values(X)

            st.subheader("SHAP Waterfall Plot")

            fig, ax = plt.subplots(figsize=(10,5))

            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=X.iloc[0]
                ),
                show=False
            )

            st.pyplot(fig)

            # =========================
            # PLAIN ENGLISH EXPLANATION
            # =========================

            st.subheader("Plain-English Explanation")

            if probability >= 0.75:

                st.error("""
                This transaction is highly suspicious because:
                - unusual transaction behavior detected
                - transaction amount appears risky
                - model identified strong fraud signals
                - device or timing pattern is abnormal
                """)

            elif probability >= 0.40:

                st.warning("""
                This transaction has moderate fraud indicators.
                Manual verification is recommended.
                """)

            else:

                st.success("""
                Transaction appears legitimate with low fraud risk.
                """)

        else:
            st.error("TransactionID not found")

    except:
        st.error("Invalid TransactionID")