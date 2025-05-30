import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load model and features
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('top8_features.pkl', 'rb') as f:
    feature_names = pickle.load(f)

st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.title("🔍 Loan Default Prediction App")
st.write("Enter the borrower’s financial details to predict loan repayment outcome.")

# Layout
col1, col2 = st.columns(2)

with col1:
    int_rate = st.slider("Interest Rate", 0.0, 1.0, 0.1, 0.01)
    log_annual_inc = st.number_input("Log Annual Income", value=10.0)
    dti = st.slider("Debt-to-Income Ratio (DTI)", 0.0, 40.0, 20.0, 0.1)
    days_with_cr_line = st.number_input("Days with Credit Line", min_value=0, value=365)

with col2:
    revol_bal = st.number_input("Revolving Balance", min_value=0.0, value=500.0)
    loan_to_income_ratio = st.slider("Loan to Income Ratio", 0.0, 1.0, 0.3, 0.01)
    revol_util = st.slider("Revolving Utilization", 0.0, 1.0, 0.5, 0.01)
    fico = st.slider("FICO Score", 300, 850, 700)

# Prepare input data using a dictionary
input_dict = {
    'int.rate': int_rate,
    'log.annual.inc': log_annual_inc,
    'dti': dti,
    'days.with.cr.line': days_with_cr_line,
    'revol.bal': revol_bal,
    'loan_to_income_ratio': loan_to_income_ratio,
    'revol.util': revol_util,
    'fico': fico
}

# Use model's expected feature order if available
expected_features = list(getattr(model, "feature_names_in_", feature_names))

# Check for missing features before prediction
missing = [feat for feat in expected_features if feat not in input_dict]
if missing:
    st.error(f"Missing features in input_dict: {missing}")
else:
    input_data = np.array([[input_dict[feat] for feat in expected_features]])
    input_df = pd.DataFrame(input_data, columns=expected_features)

    # Predict
    if st.button("🔮 Predict Loan Outcome"):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.markdown("---")
        if prediction == 1:
            st.error("💥 The loan is **likely NOT to be fully paid**.")
        else:
            st.success("✅ The loan is **likely to be fully paid**.")

        st.metric("🟩 Probability of Full Payment", f"{proba[0]*100:.2f}%")
        st.metric("🟥 Probability of Default", f"{proba[1]*100:.2f}%")

        # Bar chart
        fig, ax = plt.subplots()
        ax.bar(["Fully Paid", "Default"], proba, color=["green", "red"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

        # Add prediction to data
        result_df = input_df.copy()
        result_df["Prediction"] = ["Not Fully Paid" if prediction == 1 else "Fully Paid"]
        result_df["Probability_Fully_Paid"] = proba[0]
        result_df["Probability_Default"] = proba[1]

        # Download button
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Result as CSV", data=csv, file_name="loan_prediction_result.csv", mime="text/csv")