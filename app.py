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
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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

# --- SCIENTIFIC PATIENT EXPLANATION GENERATOR ---
def generate_patient_explanation(risk_level, systolic, diastolic, bmi, smoking, activity, family_history, cholesterol):
    """
    Generates a plain-language, scientifically grounded clinical explanation 
    based on WHO and ACC/AHA hypertension management guidelines.
    """
    explanation = []
    
    # 1. Overall Summary
    explanation.append(
        f"<b>Overall Clinical Summary:</b> Based on your submitted health parameters, the risk model classifies your profile as <b>{risk_level}</b>. "
        "This indicates your multi-factor profile shows characteristics associated with increased cardiovascular workload over time."
    )
    
    # 2. Blood Pressure Evaluation (ACC/AHA Guidelines)
    if systolic >= 140 or diastolic >= 90:
        bp_status = "Stage 2 High Blood Pressure (Hypertension)"
        bp_note = "Arterial walls experience persistent high pressure, increasing long-term cardiovascular burden."
    elif systolic >= 130 or diastolic >= 80:
        bp_status = "Stage 1 High Blood Pressure"
        bp_note = "Resting blood pressure is elevated above optimal thresholds, warranting monitoring and targeted lifestyle adjustments."
    elif systolic >= 120 and diastolic < 80:
        bp_status = "Elevated Blood Pressure"
        bp_note = "Readings are slightly above normal, though not officially categorized as clinical hypertension."
    else:
        bp_status = "Normal / Optimal Range"
        bp_note = "Resting blood pressure is within normal physiological limits (<120/80 mmHg)."
        
    explanation.append(f"• <b>Blood Pressure ({systolic}/{diastolic} mmHg):</b> Categorized as <i>{bp_status}</i>. {bp_note}")
    
    # 3. BMI & Lipid Metrics
    if bmi >= 30:
        explanation.append(f"• <b>Body Mass Index ({bmi} kg/m²):</b> Indicates obesity range. Higher body mass elevates vascular resistance, requiring the heart to exert greater force.")
    elif bmi >= 25:
        explanation.append(f"• <b>Body Mass Index ({bmi} kg/m²):</b> Indicates overweight range, a known contributing factor to baseline blood pressure elevation.")
    else:
        explanation.append(f"• <b>Body Mass Index ({bmi} kg/m²):</b> Within normal body weight range (18.5–24.9 kg/m²).")
        
    # 4. Lifestyle & Genetic Factors
    contributing = []
    if smoking == "Yes":
        contributing.append("<b>Tobacco exposure</b> (nicotine causes acute arterial constriction and accelerates endothelial inflammation)")
    if activity == "Low":
        contributing.append("<b>Low physical activity</b> (regular aerobic exercise reduces systemic vascular resistance)")
    if family_history == "Yes":
        contributing.append("<b>Family history</b> (genetic predisposition accounts for significant variance in primary hypertension susceptibility)")
    if cholesterol >= 200:
        contributing.append(f"<b>Elevated cholesterol ({cholesterol} mg/dL)</b> (lipid accumulation contributes to arterial stiffness)")
        
    if contributing:
        explanation.append("• <b>Primary Contributing Risk Factors:</b><br/> &nbsp;&nbsp;&nbsp;&ndash; " + "<br/> &nbsp;&nbsp;&nbsp;&ndash; ".join(contributing))
        
    # 5. Scientific Recommendation
    explanation.append(
        "<b>Medical Disclaimer & Recommended Actions:</b> This automated summary uses machine learning for health risk stratification and is <b>not a clinical diagnosis</b>. "
        "It is strongly recommended to share this report with a qualified healthcare professional for formal screening, ambulatory monitoring, and personalized clinical guidance."
    )
    
    return "<br/><br/>".join(explanation)


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

        # Generate Plain-Language Medical Explanation
        scientific_explanation = generate_patient_explanation(
            risk_level=risk_level,
            systolic=systolic_bp,
            diastolic=diastolic_bp,
            bmi=bmi,
            smoking=smoking_status,
            activity=physical_activity,
            family_history=family_history,
            cholesterol=cholesterol
        )

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧾 Prediction Results")
        st.markdown(f"<p class='risk-label'>Predicted Hypertension Risk: {risk_level}</p>", unsafe_allow_html=True)
        
        st.markdown("#### 🔢 Prediction Probabilities")
        st.bar_chart(pd.DataFrame({
            "Risk Level": ["Low", "Moderate", "High"],
            "Probability": probability
        }).set_index("Risk Level"))

        # Explanation Section (On-Screen)
        st.markdown("### 📋 Clinical Interpretation & Guidance")
        st.write(scientific_explanation.replace("<br/>", "\n").replace("<b>", "**").replace("</b>", "**").replace("<i>", "*").replace("</i>", "*"), unsafe_allow_html=True)

        # Feature Importance Section
        st.markdown("### 🧠 Feature Impact Analysis")
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
        except Exception:
            st.warning("⚠️ SHAP explanation visual unavailable.")

        # PDF Generation Engine
        def create_pdf():
            pdf_path = "Hypertension_Report.pdf"
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=A4,
                rightMargin=36,
                leftMargin=36,
                topMargin=36,
                bottomMargin=36
            )
            styles = getSampleStyleSheet()
            
            # Custom Paragraph Styles
            title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor("#2C3E50"), spaceAfter=12)
            section_heading = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontSize=13, leading=16, textColor=colors.HexColor("#16A085"), spaceBefore=10, spaceAfter=6)
            body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontSize=9.5, leading=13, textColor=colors.HexColor("#333333"))
            
            elements = []
            
            # Header
            elements.append(Paragraph("🩺 Hypertension Risk Assessment Report", title_style))
            elements.append(Paragraph("<b>Generated by AI Health Screening Tool</b>", body_style))
            elements.append(Spacer(1, 10))
            
            # Prediction Summary Box
            elements.append(Paragraph("<b>1. Prediction Summary</b>", section_heading))
            elements.append(Paragraph(f"<b>Assessed Risk Level:</b> {risk_level}", body_style))
            elements.append(Spacer(1, 8))
            
            # Probability Table
            prob_data = [
                ["Risk Category", "Low Risk", "Moderate Risk", "High Risk"],
                ["Estimated Probability", f"{probability[0]*100:.1f}%", f"{probability[1]*100:.1f}%", f"{probability[2]*100:.1f}%"]
            ]
            prob_table = Table(prob_data, colWidths=[130, 110, 110, 110])
            prob_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ]))
            elements.append(prob_table)
            elements.append(Spacer(1, 12))
            
            # Scientific Explanation Section
            elements.append(Paragraph("<b>2. Clinical Interpretation & Evidence-Based Insights</b>", section_heading))
            elements.append(Paragraph(scientific_explanation, body_style))
            elements.append(Spacer(1, 12))
            
            # Patient Clinical Parameters Table
            elements.append(Paragraph("<b>3. Input Clinical Metrics</b>", section_heading))
            param_data = [
                ["Parameter", "Value", "Parameter", "Value"],
                ["Age", f"{age} yrs", "Cholesterol", f"{cholesterol} mg/dL"],
                ["Gender", f"{gender}", "Smoking Status", f"{smoking_status}"],
                ["BMI", f"{bmi} kg/m²", "Alcohol Use", f"{alcohol}"],
                ["Systolic BP", f"{systolic_bp} mmHg", "Physical Activity", f"{physical_activity}"],
                ["Diastolic BP", f"{diastolic_bp} mmHg", "Family History", f"{family_history}"]
            ]
            param_table = Table(param_data, colWidths=[120, 120, 120, 120])
            param_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#ECF0F1")),
                ('BACKGROUND', (2, 0), (3, 0), colors.HexColor("#ECF0F1")),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8.5),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(param_table)
            
            doc.build(elements)
            return pdf_path

        pdf_file = create_pdf()
        with open(pdf_file, "rb") as f:
            st.download_button("📥 Download Comprehensive Patient Report (PDF)", f, file_name="Hypertension_Patient_Report.pdf")

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
    st.dataframe(
        results_df.style.background_gradient(cmap="Blues", subset=numeric_cols).format("{:.3f}", subset=numeric_cols),
        use_container_width=True
    )

# ================================================================
# --- PAGE 3: ABOUT THE TOOL (With Feedback Form) ---
# ================================================================
elif page == "💡 About this Tool":
    st.markdown("<h1 class='main-title'>💡 About this Tool</h1>", unsafe_allow_html=True)
    st.markdown("""
    This dashboard was developed as part of a research project to predict the **risk level of hypertension among Ghanaians** using advanced **machine learning algorithms** that analyze health and lifestyle factors such as **age, BMI, blood pressure, cholesterol, smoking habits, and family history**.  

    The project demonstrates how **data-driven intelligence** can empower healthcare systems, improve preventive screening, and promote public health across Ghana.  
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
