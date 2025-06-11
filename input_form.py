import streamlit as st
import pandas as pd

def get_manual_input_form(label_encoders):
    df = None
    with st.form("manual_input_form"):
        try:
            loan_id = st.text_input("Loan ID")
            no_of_dependents = st.number_input("No. of Dependents", min_value=0, max_value=10, step=1)
            education = st.selectbox("Education", label_encoders["education"].classes_.tolist())
            self_employed = st.selectbox("Self Employed", label_encoders["self_employed"].classes_.tolist())
            income_annum = st.number_input("Annual Income", min_value=0)
            loan_amount = st.number_input("Loan Amount", min_value=0)
            loan_term = st.number_input("Loan Term (months)", min_value=1)
            cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900)
            residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
            commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
            luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
            bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

            submitted = st.form_submit_button("Submit Input")
        except Exception as e:
            st.error(f"⚠️ Input error: {e}")
            return None

        if submitted:
            try:
                df = pd.DataFrame([{
                    "loan_id": loan_id,
                    "no_of_dependents": no_of_dependents,
                    "education": label_encoders["education"].transform([education])[0],
                    "self_employed": label_encoders["self_employed"].transform([self_employed])[0],
                    "income_annum": income_annum,
                    "loan_amount": loan_amount,
                    "loan_term": loan_term,
                    "cibil_score": cibil_score,
                    "residential_assets_value": residential_assets_value,
                    "commercial_assets_value": commercial_assets_value,
                    "luxury_assets_value": luxury_assets_value,
                    "bank_asset_value": bank_asset_value
                }])
                return df
            except Exception as e:
                st.error(f"⚠️ Form submission error: {e}")
    return None
