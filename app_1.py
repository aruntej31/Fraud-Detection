import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Fraud Operations Dashboard",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv("fraud_data.csv")

df = load_data()

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# =====================================================
# LOAD SHAP EXPLAINER
# =====================================================

@st.cache_resource
def load_explainer():
    return joblib.load("shap_explainer.pkl")

explainer = load_explainer()

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Overview", "Transaction Explorer", "SHAP Explainer"]
)

# =====================================================
# OVERVIEW PAGE
# =====================================================

if page == "Overview":

    st.title("📊 Fraud Detection Overview")

    # Metrics
    total_transactions = len(df)

    fraud_count = df["isFraud"].sum()

    detection_rate = (
        fraud_count / total_transactions
    ) * 100

    avg_fraud_amt = df[
        df["isFraud"] == 1
    ]["TransactionAmt"].mean()

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

    # =================================================
    # PIE CHART
    # =================================================

    st.subheader("Fraud Distribution")

    fraud_dist = (
        df["isFraud"]
        .value_counts()
        .reset_index()
    )

    fraud_dist.columns = [
        "Fraud",
        "Count"
    ]

    fraud_dist["Fraud"] = fraud_dist[
        "Fraud"
    ].map({
        0: "Non-Fraud",
        1: "Fraud"
    })

    fig = px.pie(
        fraud_dist,
        names="Fraud",
        values="Count",
        hole=0.4
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =================================================
    # HISTOGRAM
    # =================================================

    st.subheader(
        "Transaction Amount Distribution"
    )

    fig2 = px.histogram(
        df,
        x="TransactionAmt",
        color="isFraud",
        nbins=50
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# =====================================================
# TRANSACTION EXPLORER PAGE
# =====================================================

elif page == "Transaction Explorer":

    st.title("🔍 Transaction Explorer")

    # Sidebar Filters
    st.sidebar.subheader("Filters")

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

    # Filter Data
    filtered_df = df[
        (df["TransactionAmt"] >= min_amount)
        &
        (df["TransactionAmt"] <= max_amount)
    ]

    # Show Data
    st.subheader("Filtered Transactions")

    st.dataframe(filtered_df)

    # =================================================
    # SEARCH TRANSACTION
    # =================================================

    st.subheader("Search Transaction")

    transaction_id = st.text_input(
        "Enter TransactionID"
    )

    if transaction_id:

        try:

            transaction_id = int(
                transaction_id
            )

            transaction = df[
                df["TransactionID"]
                == transaction_id
            ]

            if not transaction.empty:

                st.write(transaction)

                # Features
                X = transaction.drop(
                    columns=[
                        "TransactionID",
                        "isFraud"
                    ],
                    errors="ignore"
                )

                # Prediction
                probability = (
                    model.predict_proba(X)[0][1]
                )

                st.subheader(
                    "Fraud Risk Score"
                )

                st.progress(
                    float(probability)
                )

                st.write(
                    f"Fraud Probability: {probability:.2f}"
                )

                # Risk Levels
                if probability >= 0.75:

                    st.error(
                        "🔴 High Risk"
                    )

                elif probability >= 0.40:

                    st.warning(
                        "🟡 Medium Risk"
                    )

                else:

                    st.success(
                        "🟢 Low Risk"
                    )

            else:

                st.error(
                    "Transaction not found"
                )

        except:

            st.error(
                "Invalid TransactionID"
            )

    # =================================================
    # SCATTER PLOT
    # =================================================

    st.subheader("Transaction Analysis")

    if "HourOfDay" in df.columns:

        fig3 = px.scatter(
            filtered_df,
            x="HourOfDay",
            y="TransactionAmt",
            color="isFraud",
            title="Transaction Amount vs Hour Of Day"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

# =====================================================
# SHAP EXPLAINER PAGE
# =====================================================

elif page == "SHAP Explainer":

    st.title("🧠 SHAP Fraud Explanation")

    transaction_id = st.text_input(
        "Enter TransactionID"
    )

    if transaction_id:

        try:

            transaction_id = int(
                transaction_id
            )

            row = df[
                df["TransactionID"]
                == transaction_id
            ]

            if len(row) > 0:

                st.subheader(
                    "Transaction Details"
                )

                st.write(row)

                # Features
                X = row.drop(
                    columns=[
                        "TransactionID",
                        "isFraud"
                    ],
                    errors="ignore"
                )

                # Prediction
                probability = (
                    model.predict_proba(X)[:,1][0]
                )

                st.subheader(
                    "Fraud Probability"
                )

                st.write(
                    f"{probability:.2f}"
                )

                # =====================================
                # SHAP VALUES
                # =====================================

                shap_values = (
                    explainer.shap_values(X)
                )

                st.subheader(
                    "SHAP Waterfall Plot"
                )

                fig, ax = plt.subplots(
                    figsize=(10, 5)
                )

                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=explainer.expected_value,
                        data=X.iloc[0]
                    ),
                    show=False
                )

                st.pyplot(fig)

                # =====================================
                # EXPLANATION
                # =====================================

                st.subheader(
                    "Plain-English Explanation"
                )

                if probability >= 0.75:

                    st.error("""
                    This transaction is highly suspicious because:
                    - unusual transaction behavior detected
                    - risky transaction amount
                    - strong fraud signals identified
                    - abnormal timing or device usage
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

                st.error(
                    "TransactionID not found"
                )

        except:

            st.error(
                "Invalid TransactionID"
            )
    