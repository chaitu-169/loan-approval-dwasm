# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from dwasm_model import predict_dwasm, evaluate_models
import warnings
warnings.filterwarnings('ignore')
from input_form import get_manual_input_form
from sklearn.metrics import accuracy_score

@st.cache_resource
def load_artifacts():
    xgb_model = joblib.load('models/xgb_model.pkl')
    gb_model = joblib.load('models/gb_model.pkl')
    rf_model = joblib.load('models/rf_model.pkl')
    lr_model = joblib.load('models/lr_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    weights = np.load('models/weights.npy')
    return xgb_model, gb_model, rf_model, lr_model, scaler, label_encoders, weights

st.title("🏦 Loan Approval Prediction using DWASM")

# Load models and preprocessing tools
xgb_model, gb_model, rf_model, lr_model, scaler, label_encoders, weights = load_artifacts()

# 📄 CSV Upload Section
uploaded_file = st.file_uploader("📄 Upload Loan CSV file for Evaluation", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    
    # Fix deprecation (applymap → apply)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    if 'loan_status' in df.columns:
        df['loan_status'] = df['loan_status'].replace({'Approved': 1, 'Rejected': 0}).astype(int)

        # Label encoding
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])

        # Scale numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('loan_status')
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        X = df.drop(columns=['loan_status'])
        y = df['loan_status']

        models = [xgb_model, gb_model, rf_model]
        accuracies = evaluate_models(models, X, y)
        dwasm_preds = predict_dwasm(models, weights, X)
        dwasm_acc = np.mean(dwasm_preds == y)
        lr_acc = accuracy_score(y, lr_model.predict(X))

        # Display results
        st.subheader("📊 Model Accuracies")
        for name, acc in accuracies.items():
            st.write(f"**{name} Accuracy**: `{acc:.4f}`")
        st.write(f"**Logistic Regression Accuracy**: `{lr_acc:.4f}`")
        st.write(f"**DWASM Ensemble Accuracy**: `🎯 {dwasm_acc:.4f}`")
    else:
        st.warning("⚠️ 'loan_status' column missing in CSV. Can't compute accuracy.")

# 📝 Manual Input Prediction Section
st.subheader("📝 Manual Input for Prediction")
manual_input_df = get_manual_input_form(label_encoders)

if manual_input_df is not None:
    # Debug input data
    st.write("### Raw Input Data (Debug):")
    st.dataframe(manual_input_df)
    
    # Make sure manual input is properly formatted
    # For categorical columns, ensure they're properly encoded
    categorical_cols = [col for col in manual_input_df.columns if manual_input_df[col].dtype == 'object']
    for col in categorical_cols:
        if col in label_encoders:
            try:
                manual_input_df[col] = label_encoders[col].transform(manual_input_df[col])
            except:
                st.warning(f"Warning: Could not transform column {col} with label encoder. This may cause errors.")
    
    # For numeric columns, ensure they're numeric
    numeric_cols = [col for col in manual_input_df.columns if col not in categorical_cols]
    for col in numeric_cols:
        manual_input_df[col] = pd.to_numeric(manual_input_df[col], errors='coerce')
        # Fill any NaN values with 0
        manual_input_df[col] = manual_input_df[col].fillna(0)
    
    # Scale numeric columns using the same scaler used for training
    try:
        input_scaled = scaler.transform(manual_input_df)
        
        # Debug scaled data
        st.write("### Scaled Input Data (Debug):")
        st.dataframe(pd.DataFrame(input_scaled, columns=manual_input_df.columns))

        # Get individual model predictions for debugging
        model_preds = []
        model_names = ["XGBoost", "Gradient Boosting", "Random Forest", "Logistic Regression"]
        
        for i, model in enumerate([xgb_model, gb_model, rf_model, lr_model]):
            pred = model.predict(input_scaled)
            model_preds.append(pred[0])
            st.write(f"**{model_names[i]} Prediction**: {'✅ Approved' if pred[0] == 1 else '❌ Rejected'}")
        
        # Predict loan status using DWASM ensemble - display individual contributions
        individual_preds = [model.predict_proba(input_scaled)[0][1] for model in [xgb_model, gb_model, rf_model]]
        
        # Display individual probabilities
        st.write("### Individual Model Probabilities (higher = more likely to approve):")
        for i, (name, prob) in enumerate(zip(["XGBoost", "Gradient Boosting", "Random Forest"], individual_preds)):
            st.write(f"**{name}**: {prob:.4f} (Weight: {weights[i]/sum(weights[:3]):.4f})")
        
        # Calculate weighted average with current weights
        weighted_prob = sum(w*p for w, p in zip(weights[:3]/sum(weights[:3]), individual_preds))
        st.write(f"**Current weighted probability**: {weighted_prob:.4f}")
        
        # Try different weight combinations
        st.write("### Testing different weight combinations:")
        # Equal weights
        equal_weights = np.ones(3)/3
        equal_weighted_prob = sum(w*p for w, p in zip(equal_weights, individual_preds))
        st.write(f"**Equal weights probability**: {equal_weighted_prob:.4f}")
        
        # De-emphasize XGBoost
        xgb_reduced_weights = np.array([0.2, 0.4, 0.4])
        xgb_reduced_prob = sum(w*p for w, p in zip(xgb_reduced_weights, individual_preds))
        st.write(f"**XGBoost de-emphasized probability**: {xgb_reduced_prob:.4f}")
        
        # Use updated weights with XGBoost de-emphasized
        prediction = predict_dwasm([xgb_model, gb_model, rf_model], xgb_reduced_weights, input_scaled)
        result = "✅ Approved" if prediction[0] == 1 else "❌ Rejected"
        st.success(f"**Corrected DWASM Ensemble Prediction:** {result}")
        
        # Show model weights
        st.write("### Model Weights in Ensemble:")
        normalized_weights = weights / np.sum(weights)
        for i, w in enumerate(normalized_weights[:3]):  # Only show the first 3 models
            st.write(f"**{model_names[i]} Weight**: {w:.4f}")

        # 🧠 DWASM Explanation using SHAP
        st.subheader("🔍 Explanation for Prediction")
    except Exception as e:
        st.error(f"Error in scaling or prediction: {str(e)}")
        st.exception(e)

    try:
        # Create input DataFrame for SHAP compatibility
        input_scaled_df = pd.DataFrame(input_scaled, columns=manual_input_df.columns)
        
        # Debug info about feature scales - handle string values
        st.write("### Feature Scale Analysis:")
        feature_ranges = {}
        for col in manual_input_df.columns:
            # Convert to numeric first, with errors='coerce' to handle strings
            numeric_vals = pd.to_numeric(manual_input_df[col], errors='coerce')
            feature_ranges[col] = {
                "min": numeric_vals.min() if not pd.isna(numeric_vals.min()) else "N/A", 
                "max": numeric_vals.max() if not pd.isna(numeric_vals.max()) else "N/A",
                "mean": numeric_vals.mean() if not pd.isna(numeric_vals.mean()) else "N/A",
                "type": manual_input_df[col].dtype
            }
        
        # Display feature with unusually large range that might dominate SHAP values
        st.dataframe(pd.DataFrame(feature_ranges).T)
        st.write("⚠️ Features with large ranges may disproportionately affect SHAP values")
        
        # Get SHAP values for each model - with feature normalization
        shap_vals_list = []
        feature_names = manual_input_df.columns
        
        # XGBoost
        explainer_xgb = shap.Explainer(xgb_model)
        shap_values_xgb = explainer_xgb(input_scaled_df)
        
        # Check dimensions and extract accordingly
        if len(shap_values_xgb.values.shape) == 3:  # Multi-class output (samples, features, classes)
            # For binary classification, index 1 is for class 1 (approval)
            shap_vals_xgb = shap_values_xgb.values[0, :, 1] if prediction[0] == 1 else shap_values_xgb.values[0, :, 0]
        else:  # Single dimension output
            shap_vals_xgb = shap_values_xgb.values[0]
        
        shap_vals_list.append(shap_vals_xgb)
        
        # Gradient Boosting
        explainer_gb = shap.Explainer(gb_model)
        shap_values_gb = explainer_gb(input_scaled_df)
        
        if len(shap_values_gb.values.shape) == 3:
            shap_vals_gb = shap_values_gb.values[0, :, 1] if prediction[0] == 1 else shap_values_gb.values[0, :, 0]
        else:
            shap_vals_gb = shap_values_gb.values[0]
            
        shap_vals_list.append(shap_vals_gb)
        
        # Random Forest
        explainer_rf = shap.Explainer(rf_model)
        shap_values_rf = explainer_rf(input_scaled_df)
        
        if len(shap_values_rf.values.shape) == 3:
            shap_vals_rf = shap_values_rf.values[0, :, 1] if prediction[0] == 1 else shap_values_rf.values[0, :, 0]
        else:
            shap_vals_rf = shap_values_rf.values[0]
            
        shap_vals_list.append(shap_vals_rf)
        
        # Check shapes before combining
        shapes = [vals.shape for vals in shap_vals_list]
        st.write(f"Debug - SHAP value shapes: {shapes}")
        
        # All shapes match, proceed with ensemble explanation - using corrected weights
        # De-emphasize XGBoost in the SHAP calculation
        xgb_reduced_weights = np.array([0.2, 0.4, 0.4])  # Same weights used for prediction
        shap_vals_combined = (
            xgb_reduced_weights[0] * shap_vals_list[0] +
            xgb_reduced_weights[1] * shap_vals_list[1] +
            xgb_reduced_weights[2] * shap_vals_list[2]
        )
        
        # Create DataFrame for visualization with normalized impact values
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_vals_combined,
            'Normalized Impact': shap_vals_combined / np.max(np.abs(shap_vals_combined)) if np.max(np.abs(shap_vals_combined)) > 0 else shap_vals_combined
        }).sort_values(by='Normalized Impact', key=abs, ascending=False)
        
        # Show top influencing features
        st.write("### 🔍 Top features influencing this decision:")
        st.dataframe(shap_df)
        
        # Plot SHAP values - using normalized impact
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if x < 0 else 'green' for x in shap_df.head(5)['Normalized Impact']]
        
        shap_df.head(5).plot.barh(
            x='Feature', 
            y='Normalized Impact', 
            ax=ax, 
            color=colors
        )
        
        if prediction[0] == 1:
            ax.set_title("DWASM Weighted SHAP Feature Contributions (Approval)")
        else:
            ax.set_title("DWASM Weighted SHAP Feature Contributions (Rejection)")
            
        ax.set_xlabel("Impact on Decision")
        ax.invert_yaxis()
        st.pyplot(fig)
        
        # Show explanation based on prediction
        if prediction[0] == 0:  # Rejected
            st.error("❗ **Reason for Rejection:**")
            # For rejection, sort by absolute value but highlight negative values
            negative_impact = shap_df.sort_values(by='Normalized Impact').head(3)
            for _, row in negative_impact.iterrows():
                st.write(f"- **{row['Feature']}** contributed to rejection (Impact: `{row['Normalized Impact']:.4f}`)")
        else:  # Approved
            st.success("✅ **Key factors for Approval:**")
            # For approval, find most positive values
            positive_impact = shap_df.sort_values(by='Normalized Impact', ascending=False).head(3)
            for _, row in positive_impact.iterrows():
                st.write(f"- **{row['Feature']}** contributed to approval (Impact: `{row['Normalized Impact']:.4f}`)")

    except Exception as e:
        st.error(f"Error calculating SHAP values: {str(e)}")
        st.write("Detailed error information:")
        st.exception(e)