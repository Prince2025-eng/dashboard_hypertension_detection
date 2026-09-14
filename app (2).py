import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
from lime.lime_tabular import LimeTabularExplainer
from streamlit_option_menu import option_menu
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import os

# --- LOAD MODEL COMPONENTS ---
best_model = joblib.load("best_model.joblib")
preprocessor = joblib.load("preprocessor.joblib")
selected_features = joblib.load("selected_features.joblib")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Hypertension Risk Detection Tool",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🧭 Navigation")
    page = option_menu(
        menu_title=None,
        options=["🏥 Hypertension Risk Detection", "📊 Model Overview", "💡 About this Tool"],
        icons=["activity", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#f8fafc"},
            "icon": {"color": "#5B86E5", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px", "--hover-color": "#e9ecef"},
            "nav-link-selected": {"background-color": "#5B86E5", "color": "white"},
        },
    )
    st.markdown("---")
    st.sidebar.info("Created with ❤️ using Streamlit and Machine Learning.")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        body { background-color: #f8fafc; color: #222; font-family: 'Helvetica Neue', sans-serif; }
        .main-title { text-align: center; color: #2c3e50; font-size: 2.4em; font-weight: 700; margin-bottom: 5px; }
        .subtitle { text-align: center; color: #7f8c8d; font-size: 1.1em; margin-bottom: 30px; }
        .stButton>button {
            width: 100%; background: linear-gradient(90deg, #36D1DC, #5B86E5);
            color: white; font-weight: 600; border-radius: 10px; padding: 10px; transition: all 0.3s ease;
        }
        .stButton>button:hover { background: linear-gradient(90deg, #5B86E5, #36D1DC); transform: scale(1.03); }
        .card { background-color: white; padding: 20px; border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-top: 20px; }
        .risk-label { font-size: 1.3em; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# ================================================================
# --- PAGE 1: HYPERTENSION RISK DETECTION ---
# ================================================================
if page == "🏥 Hypertension Risk Detection":
    st.markdown("<h1 class='main-title'>🩺 Hypertension Risk Detection Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>An AI-powered tool to estimate hypertension risk using patient health indicators.</p>", unsafe_allow_html=True)

    st.subheader("🧍‍♂️ Enter Patient Clinical Information")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        occupation = st.text_input("Occupation", "Teacher")
    with col2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.5)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=250, value=120)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
    with col3:
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
        smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
        alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "High"])
        physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
        family_history = st.selectbox("Family History of Hypertension", ["Yes", "No"])

    # --- PREDICTION SECTION ---
    if st.button("🔍 Predict Risk Level"):
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Occupation': [occupation],
            'BMI': [bmi],
            'Systolic_BP': [systolic_bp],
            'Diastolic_BP': [diastolic_bp],
            'Cholesterol': [cholesterol],
            'Smoking_Status': [smoking_status],
            'Alcohol_Consumption': [alcohol],
            'Physical_Activity_Level': [physical_activity],
            'Family_History': [family_history]
        })

        X_new = preprocessor.transform(input_data)
        X_new_sel = X_new[:, selected_features]
        prediction = best_model.predict(X_new_sel)[0]
        probability = best_model.predict_proba(X_new_sel)[0]

        risk_map = {0: "🟢 Low Risk", 1: "🟡 Moderate Risk", 2: "🔴 High Risk"}
        risk_level = risk_map[prediction]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧾 Prediction Results")
        st.markdown(f"<p class='risk-label'>Predicted Hypertension Risk: {risk_level}</p>", unsafe_allow_html=True)
        st.markdown("#### 🔢 Prediction Probabilities")
        st.bar_chart(pd.DataFrame({
            "Risk Level": ["Low", "Moderate", "High"],
            "Probability": probability
        }).set_index("Risk Level"))

        # Explanation Section
        st.markdown("### 🧠 Explanation of Results")
        explanation_text = ""
        try:
            explainer = shap.Explainer(best_model, X_new_sel)
            shap_values = explainer(X_new_sel)
            shap_df = pd.DataFrame({
                "Feature": selected_features,
                "Input Value": np.round(X_new_sel[0], 3),
                "SHAP Value": np.round(shap_values.values[0], 4)
            }).sort_values(by="SHAP Value", ascending=False)
            fig_shap = px.bar(
                shap_df.sort_values(by="SHAP Value", ascending=True),
                x="SHAP Value", y="Feature", orientation="h",
                color="SHAP Value", color_continuous_scale="RdBu",
                title="Feature Influence on Risk Prediction"
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.info("The above chart shows how each factor influenced your hypertension risk level.")
        except Exception:
            st.warning("⚠️ SHAP explanation unavailable — using LIME instead.")

        # PDF Report
        def create_pdf():
            pdf_path = "Hypertension_Report.pdf"
            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []
            elements.append(Paragraph("🩺 Hypertension Risk Prediction Report", styles["Title"]))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"<b>Predicted Risk:</b> {risk_level}", styles["Normal"]))
            elements.append(Paragraph(f"<b>Explanation:</b> {explanation_text}", styles["Normal"]))
            elements.append(Spacer(1, 12))
            prob_table = Table([["Low", "Moderate", "High"], [f"{probability[0]:.2f}", f"{probability[1]:.2f}", f"{probability[2]:.2f}"]])
            prob_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER')
            ]))
            elements.append(prob_table)
            elements.append(Spacer(1, 12))
            for key, value in input_data.iloc[0].items():
                elements.append(Paragraph(f"{key}: {value}", styles["Normal"]))
            doc.build(elements)
            return pdf_path

        pdf_file = create_pdf()
        with open(pdf_file, "rb") as f:
            st.download_button("📥 Download Prediction Report (PDF)", f, file_name="Hypertension_Report.pdf")

        st.markdown("</div>", unsafe_allow_html=True)

# ================================================================
# --- PAGE 2: MODEL OVERVIEW ---
# ================================================================
elif page == "📊 Model Overview":
    st.markdown("<h1 class='main-title'>📊 Model Performance Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Summary of machine learning models developed and evaluated for hypertension risk prediction.</p>", unsafe_allow_html=True)
    st.markdown("""
    - **Algorithms Used:** Logistic Regression, Support Vector Machine (SVM), Neural Network (MLPClassifier)
    - **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, and ROC-AUC  
    - **Best Performing Model:** 🧠 Neural Network (ROC-AUC = 0.88, Accuracy = 0.76)
    """)
    results_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Support Vector Machine", "Neural Network"],
        "Accuracy": [0.720, 0.755, 0.763],
        "Precision": [0.750, 0.759, 0.768],
        "Recall": [0.720, 0.755, 0.763],
        "F1-Score": [0.693, 0.752, 0.760],
        "ROC-AUC": [0.877, 0.880, 0.882]
    })
    numeric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    st.markdown("### 📈 Model Performance Table")
    st.dataframe(results_df.style.background_gradient(cmap="Blues", subset=numeric_cols).format("{:.3f}"), use_container_width=True)

# ================================================================
# --- PAGE 3: ABOUT THE TOOL (With Feedback Form) ---
# ================================================================
elif page == "💡 About this Tool":
    st.markdown("<h1 class='main-title'>💡 About this Tool</h1>", unsafe_allow_html=True)
    st.markdown("""
    This dashboard was developed by **Prince Amponsah**, a passionate **Data Science student**,  
    as part of his **final-year thesis project**.  

    The tool aims to predict the **risk level of hypertension among Ghanaians** using advanced  
    **machine learning algorithms** that analyze health and lifestyle factors such as **age, BMI,  
    blood pressure, cholesterol, smoking habits, and family history**.  

    The project demonstrates how **data-driven intelligence** can empower healthcare systems,  
    improve preventive screening, and promote public health across Ghana.  
    """)
    st.info("🩺 Empowering healthcare through data science and machine learning.")

    st.markdown("---")
    st.subheader("💬 Share Your Feedback")

    st.markdown("We value your input! Please share your feedback or suggestions to help us improve this dashboard.")

    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email (optional)")
        feedback_text = st.text_area("Your Feedback", placeholder="Type your comments or suggestions here...")
        submitted = st.form_submit_button("📩 Submit Feedback")

        if submitted:
            if feedback_text.strip() == "":
                st.warning("⚠️ Please write some feedback before submitting.")
            else:
                feedback_entry = pd.DataFrame({
                    "Name": [name],
                    "Email": [email],
                    "Feedback": [feedback_text],
                    "Timestamp": [pd.Timestamp.now()]
                })
                try:
                    file_exists = os.path.isfile("user_feedback.csv")
                    feedback_entry.to_csv("user_feedback.csv", mode="a", header=not file_exists, index=False)
                    st.success("✅ Thank you for your feedback! It has been recorded successfully.")
                except Exception as e:
                    st.error(f"⚠️ Could not save feedback due to an error: {e}")
