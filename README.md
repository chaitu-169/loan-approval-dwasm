# 🏦 Loan Approval Prediction using DWASM

This project implements a **loan approval prediction system** using a custom ensemble model called **DWASM** (Dynamic Weighted Adaptive Stacking Model). It predicts whether a loan application will be approved or rejected, based on applicant financial data.

## 📌 Project Highlights

- Built using **tree-based models**: XGBoost, Random Forest, and Gradient Boosting
- DWASM combines base model predictions using dynamic weighted logic
- Interactive **Streamlit** app for user input and live prediction
- Includes **SHAP explainability** to visualize feature impact
- Trained on real-world-style structured dataset

---

## 📁 Project Structure
Loan_Approval-DWASM/
├── app.py/
│── models 
│  ├── rf_model.pkl
│  ├── gb_model.pkl  
│  ├── xgb_model.pkl
│  ├── scaler.pkl
│  ├── lr_model.pkl
│  ├── weights.npy
│  └── label_encoders.pkl
├── loan_data.csv # Sample loan data
├── input_form.py
├── dwasm_model.py
├── requirements.txt
└── README.md # This file

---

## ⚙️ Setup Instructions

### 🔧 Requirements

Install dependencies using:
```bash
pip install -r requirements.txt

🚀 How to Run
Navigate to the project root directory and launch the app:

streamlit run app/loan_approval_app.py
The app will open in your browser. Enter loan details in the form to get a prediction.

🧮 Input Fields

Loan ID

Number of Dependents

Education

Self Employed

Annual Income

Loan Amount

Loan Term (months)

CIBIL Score (300–900)

Residential Assets Value

Commercial Assets Value

Luxury Assets Value

Bank Asset Value

✅ Output
Prediction: Loan Approved / Rejected

Explainability: SHAP feature contribution plot

📊 ML Models Used
All base models are tree-based classifiers:

XGBoostClassifier

RandomForestClassifier

GradientBoostingClassifier

The DWASM ensemble assigns dynamic weights to each model based on real-time accuracy and prediction strength.

📄 Authors
Seelam Sriram chaitanya


For any Queries
Contact-sriramchaitu383@gmail.com

