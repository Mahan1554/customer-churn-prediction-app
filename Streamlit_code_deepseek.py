import streamlit as st
import pandas as pd
import pickle
import numpy as np
from lifelines import CoxPHFitter
from sksurv.util import Surv

class ChurnPreprocessor:
    def __init__(self, encoder_path, scaler_path, model_path, survival_model_path, encoded_columns_path, feature_names_path):
        # Load saved encoder, scaler, models, and feature metadata
        with open(encoder_path, "rb") as f:
            self.ohe = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(survival_model_path, "rb") as f:
            self.survival_model = pickle.load(f)
        with open(encoded_columns_path, "rb") as f:
            self.encoded_columns = pickle.load(f)
        with open(feature_names_path, "rb") as f:
            self.feature_names = pickle.load(f)

        # Define feature types
        self.numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_features = list(self.ohe.feature_names_in_)

    def preprocess(self, customer_data):
        # Extract customer ID
        customer_id = customer_data.pop("customerID", "Unknown")

        # Convert to DataFrame
        new_data = pd.DataFrame([customer_data])

        # Ensure `SeniorCitizen` is mapped correctly
        new_data["SeniorCitizen"] = new_data["SeniorCitizen"].map({0: "No", 1: "Yes"})

        # Ensure correct data types
        new_data[self.numerical_features] = new_data[self.numerical_features].astype(float)
        new_data[self.categorical_features] = new_data[self.categorical_features].astype(str)

        # One-hot encode categorical features
        encoded_data = self.ohe.transform(new_data[self.categorical_features])
        encoded_df = pd.DataFrame(encoded_data, columns=self.ohe.get_feature_names_out())

        # Combine with numerical features
        new_numerical = new_data[self.numerical_features].reset_index(drop=True)
        new_processed = pd.concat([new_numerical, encoded_df], axis=1)

        # Ensure correct feature order & add missing columns
        new_processed = new_processed.reindex(columns=self.feature_names, fill_value=0)

        # Scale numerical columns
        new_processed[self.numerical_features] = self.scaler.transform(new_processed[self.numerical_features])

        return customer_id, new_processed

    def get_hazard_ratio_table(self):
        """Returns a formatted DataFrame of hazard ratios and interpretations."""
        hazard_ratios = np.exp(self.survival_model.params_)

        # Create interpretation strings
        interpretations = []
        for hr in hazard_ratios:
            if hr > 1:
                interpretations.append(f"{hr:.1f}x higher churn risk")
            elif hr < 1:
                interpretations.append(f"{1/hr:.1f}x lower risk (protective)")
            else:
                interpretations.append("No effect")

        # Build DataFrame
        hr_table = pd.DataFrame({
            "Feature": hazard_ratios.index,
            "Hazard Ratio": hazard_ratios.values,
            "Interpretation": interpretations
        })

        return hr_table.sort_values("Hazard Ratio", ascending=False)

    def predict(self, customer_data):
        customer_id = customer_data.get("customerID", "Unknown")

        # Preprocess input data and unpack tuple
        customer_id, new_processed = self.preprocess(customer_data)

        # Keep DataFrame version for survival model (Cox PH needs column names)
        new_processed_df = new_processed.copy()

        # Convert to NumPy array for XGBoost
        new_processed_array = new_processed.values

        # Predict classification
        churn_prediction = self.model.predict(new_processed_array)[0]
        churn_probability = self.model.predict_proba(new_processed_array)[0][1]

        # Predict survival time using DataFrame (Cox PH expects named columns)
        survival_time = self.survival_model.predict_median(new_processed_df)

        # Get hazard ratios
        hazard_ratios = np.exp(self.survival_model.params_)

        # Find top churn reasons (highest hazard ratio features)
        customer_features = pd.Series(new_processed_array[0], index=self.feature_names)
        feature_impact = customer_features * hazard_ratios
        top_churn_reasons = feature_impact.nlargest(5).index.tolist()

        # Output Prediction
        if churn_prediction == 0:
            return f"🔍 **Expected Customer Behavior for Customer ID: {customer_id}**\n✅ **Not likely to churn**"
        else:
            if np.isinf(survival_time):
                return f"🔍 **Expected Customer Behavior for Customer ID: {customer_id}**\n✅ **Not likely to churn**"
            else:
                reasons = "\n".join([f"🔸 {reason.replace('_', ': ')}" for reason in top_churn_reasons])
                return (
                    f"🔍 **Expected Customer Behavior for Customer ID: {customer_id}**\n"
                    f"⚠️ **Likely to churn in {survival_time:.2f} months**\n\n"
                    f"📌 **Top Reasons for Churn:**\n"
                    f"{reasons}"
                )

# Load preprocessor with caching
@st.cache_resource
def load_preprocessor():
        return ChurnPreprocessor(
        r"C:\Users\mahan\Downloads\Play\Projects\Churn Prediction\encoder_churn.pkl",
        r"C:\Users\mahan\Downloads\Play\Projects\Churn Prediction\scaler_churn.pkl",
        r"C:\Users\mahan\Downloads\Play\Projects\Churn Prediction\best_model_churn.pkl",
        r"C:\Users\mahan\Downloads\Play\Projects\Churn Prediction\cox_model.pkl",
        r"C:\Users\mahan\Downloads\Play\Projects\Churn Prediction\encoded_columns.pkl",
        r"C:\Users\mahan\Downloads\Play\Projects\Churn Prediction\feature_names.pkl"
    )


preprocessor = load_preprocessor()

# Custom CSS styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5em;
        color: #2E86C1;
        text-align: center;
        padding: 20px;
        margin-bottom: 30px;
    }
    .result-section {
        padding: 20px;
        background-color: #F8F9F9;
        border-radius: 10px;
        margin: 20px 0;
    }
    .analysis-section {
        margin: 30px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar - Customer Inputs
st.sidebar.header("Customer Details")
customer_id = st.sidebar.text_input("Customer ID (optional)")

# Senior Citizen mapping
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
senior_citizen = 0 if senior_citizen == "No" else 1

# Numerical features
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=0)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=0.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=0.0)

# Collect categorical features
customer_data = {
    "customerID": customer_id,
    "SeniorCitizen": senior_citizen,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

feature_names_in = preprocessor.ohe.feature_names_in_.tolist()
for feature in preprocessor.categorical_features:
    if feature == "SeniorCitizen":
        continue
    idx = feature_names_in.index(feature)
    categories = preprocessor.ohe.categories_[idx].tolist()
    customer_data[feature] = st.sidebar.selectbox(feature, categories)

# Prediction button
predict_button = st.sidebar.button("🔮 Predict Churn")

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("### How to Use")
st.sidebar.markdown("""
1. Fill customer details in the sidebar
2. Click 'Predict' for churn prediction
3. Click analysis buttons below for insights
""")

# Main Page
st.markdown('<h1 class="main-title">📈 Customer Churn Prediction Analyzer</h1>', unsafe_allow_html=True)

# Analysis Buttons
col1, col2, col3 = st.columns(3)
with col1:
    protective_btn = st.button("🛡️ Protective Factors")
with col2:
    risk_btn = st.button("🔥 Risk Factors")
with col3:
    hazard_btn = st.button("📊 Hazard Analysis")

# Handle analysis buttons
if protective_btn:
    hr_table = preprocessor.get_hazard_ratio_table()
    retention = hr_table[hr_table["Hazard Ratio"] < 1]
    st.markdown("### Top Retention Drivers")
    st.dataframe(retention[["Feature", "Hazard Ratio"]], hide_index=True)

if risk_btn:
    hr_table = preprocessor.get_hazard_ratio_table()
    risk = hr_table[hr_table["Hazard Ratio"] > 1]
    st.markdown("### Top Churn Risk Factors")
    st.dataframe(risk[["Feature", "Hazard Ratio"]], hide_index=True)

if hazard_btn:
    hr_table = preprocessor.get_hazard_ratio_table()
    st.markdown("### Full Hazard Ratio Analysis")
    st.dataframe(hr_table, hide_index=True)

# Handle prediction
if predict_button:
    result = preprocessor.predict(customer_data)
    
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    
    if "⚠️" in result:
        parts = result.split("\n")
        st.markdown(f"**{parts[0]}**")
        st.warning(parts[1])
        
        if len(parts) > 3:  # Ensure we have reasons to display
            st.markdown("**Top Churn Reasons**")
            for line in parts[3:]:
                if line.strip():  # Only process non-empty lines
                    st.markdown(f"- {line.replace('🔸 ', '')}")
    else:
        parts = result.split("\n")
        st.markdown(f"**{parts[0]}**")
        st.success(parts[1])
    
    st.markdown('</div>', unsafe_allow_html=True)