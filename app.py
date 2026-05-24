import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Fraud Operations Dashboard",
    page_icon="📊",
    layout="wide"
)

# Main title
st.title("📊 Fraud Operations Dashboard")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Transaction Explorer", "SHAP Explainer"]
)

# Pages
if page == "Overview":
    st.header("Overview")
    st.write("Welcome to the Fraud Detection System.")

elif page == "Transaction Explorer":
    st.header("Transaction Explorer")
    st.write("Search and explore transactions here.")

elif page == "SHAP Explainer":
    st.header("SHAP Explainer")
    st.write("Explain fraud predictions using SHAP values.")