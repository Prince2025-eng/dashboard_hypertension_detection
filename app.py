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
    page_title="Hypertension Risk Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODERN CUSTOM CSS STYLING ---
st.markdown("""
    <style>
        /* Main background and typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background-color: #f8fafc;
        }

        /* Top Header Container */
        .header-box {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 2.2rem 2rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
        }
        
        .header-box h1 {
            color: #ffffff !important;
            font-size: 2.3em !important;
            font-weight: 700 !important;
            margin-bottom: 8px !important;
        }

        .header-box p {
            color: #94a3b8 !important;
            font-size: 1.05em !important;
            margin: 0 !important;
        }

        /* Clean Card Layouts */
        .custom-card {
            background-color: #ffffff;
            border-radius: 14px;
            padding: 22px 26px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }

        /* Dynamic Risk Banner Cards */
        .risk-banner-high {
            background: linear-gradient(135deg, #fef2f2 0%, #ffe4e6 100%);
            border-left: 6px solid #ef4444;
            padding: 18px 22px;
            border-radius: 12px;
            color: #991b1b;
            font-weight: 700;
            font-size: 1.35em;
            margin-bottom: 15px;
        }
        
        .risk-banner-moderate {
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border-left: 6px solid #f59e0b;
            padding: 18px 22px;
            border-radius: 12px;
            color: #92400e;
            font-weight: 700;
            font-size: 1.35em;
            margin-bottom: 15px;
        }

        .risk-banner-low {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border-left: 6px solid #10b981;
            padding: 18px 22px;
            border-radius: 12px;
            color: #065f46;
            font-weight: 700;
            font-size: 1.35em;
            margin-bottom: 15px;
        }

        /* Metric Pill Displays */
        .metric-pill {
            background-color: #f1f5f9;
            border-radius: 10px;
            padding: 12px 16px;
            text-align: center;
            border: 1px solid #e2e8f0;
        }
        .metric-pill .val {
            font-size: 1.3em;
            font-weight: 700;
            color: #0f172a;
        }
        .metric-pill .lbl {
            font-size: 0.82em;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Action Buttons */
        .stButton>button {
            width: 100%;
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: white;
            font-weight: 600;
            font-size: 1.05em;
            border-radius: 10px;
            padding: 12px 20px;
            border: none;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
            transition: all 0.25s ease;
        }
        
        .stButton>button:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
            transform: translateY(-1px);
        }

        /* Form section headings */
        .section-label {
            font-size: 1.1em;
            font-weight: 600;
            color: #334155;
            margin-bottom: 12px;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 6px;
        }
    </style>
""", unsafe_allow_html=True)


# --- SCIENTIFIC PATIENT EXPLANATION GENERATOR ---
def generate_patient_explanation(risk_level, systolic, diastolic, bmi, smoking, activity, family_history, cholesterol):
    explanation = []
    
    explanation.append(
        f"<b>Overall Clinical Summary:</b> Based on your submitted health parameters, the risk model classifies your profile as <b>{risk_level}</b>. "
        "This indicates your multi-factor profile shows characteristics associated with increased cardiovascular workload over time."
    )
    
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
    
    if bmi >= 30:
        explanation.append(f"• <b>Body Mass Index ({bmi} kg/m²):</b> Indicates obesity range. Higher body mass elevates vascular resistance, requiring the heart to exert greater force.")
    elif bmi >= 25:
        explanation.append(f"• <b>Body Mass Index ({bmi} kg/m²):</b> Indicates overweight range, a known contributing factor to baseline blood pressure elevation.")
    else:
        explanation.append(f"• <b>Body Mass Index ({bmi} kg/m²):</b> Within normal body weight range (18.5–24.9 kg/m²).")
        
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
        
    explanation.append(
        "<b>Medical Disclaimer & Recommended Actions:</b> This automated summary uses machine learning for health risk stratification and is <b>not a clinical diagnosis</b>. "
        "It is strongly recommended to share this report with a qualified healthcare professional for formal screening, ambulatory monitoring, and personalized clinical guidance."
    )
    
    return "<br/><br/>".join(explanation)


# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387581.png", width=70)
    st.markdown("### **Hypertension AI**")
    page = option_menu(
        menu_title=None,
        options=["🏥 Risk Detection", "📊 Model Intelligence", "💡 Project Overview"],
        icons=["activity", "cpu", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "icon": {"color": "#2563eb", "font-size": "18px"},
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "4px", "border-radius": "8px"},
            "nav-link-selected": {"background-color": "#2563eb", "color": "white"},
        },
    )
    st.markdown("---")
    st.caption("🩺 **Clinical Decision Support System**")
    st.caption("Powered by Machine Learning & Predictive Analytics.")

# ================================================================
# --- PAGE 1: HYPERTENSION RISK DETECTION ---
# ================================================================
if page == "🏥 Risk Detection":
    st.markdown("""
        <div class='header-box'>
            <h1>🩺 Hypertension Risk Assessment Tool</h1>
            <p>Enter patient physiological parameters to perform automated clinical stratification.</p>
        </div>
    """, unsafe_allow_html=True)

    # Input Form Layout
    with st.container():
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>📋 Patient Clinical Parameters</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (Years)", min_value=10, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            occupation = st.text_input("Occupation", "Teacher")
        with col2:
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=24.5)
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=250, value=120)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
        with col3:
            cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
            smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
            alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "High"])
            physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
            family_history = st.selectbox("Family History of Hypertension", ["Yes", "No"])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ Calculate Risk & Generate Report")
        st.markdown("</div>", unsafe_allow_html=True)

    # Quick Metrics Display Bar
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='metric-pill'><div class='val'>{systolic_bp}/{diastolic_bp}</div><div class='lbl'>Blood Pressure</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-pill'><div class='val'>{bmi}</div><div class='lbl'>Body Mass Index</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-pill'><div class='val'>{cholesterol} mg/dL</div><div class='lbl'>Cholesterol</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='metric-pill'><div class='val'>{family_history}</div><div class='lbl'>Family History</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- PREDICTION SECTION ---
    if predict_btn:
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

        risk_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
        risk_level = risk_map[prediction]

        # Styled Risk Output Banners
        if prediction == 2:
            st.markdown(f"<div class='risk-banner-high'>🔴 Predicted Category: High Risk</div>", unsafe_allow_html=True)
        elif prediction == 1:
            st.markdown(f"<div class='risk-banner-moderate'>🟡 Predicted Category: Moderate Risk</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='risk-banner-low'>🟢 Predicted Category: Low Risk</div>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1])

        # Interactive Chart
        with col_left:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>📊 Model Risk Probability Distribution</div>", unsafe_allow_html=True)
            prob_df = pd.DataFrame({
                "Risk Category": ["Low Risk", "Moderate Risk", "High Risk"],
                "Probability": probability
            })
            fig_prob = px.bar(
                prob_df, 
                x="Risk Category", 
                y="Probability", 
                color="Risk Category",
                color_discrete_sequence=["#10b981", "#f59e0b", "#ef4444"],
                text_auto='.1%'
            )
            fig_prob.update_layout(
                showlegend=False, 
                height=320, 
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Explainability SHAP Chart
        with col_right:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>🧠 Feature Contribution (SHAP Analysis)</div>", unsafe_allow_html=True)
            try:
                explainer = shap.Explainer(best_model, X_new_sel)
                shap_values = explainer(X_new_sel)
                shap_df = pd.DataFrame({
                    "Feature": selected_features,
                    "SHAP Value": np.round(shap_values.values[0], 4)
                }).sort_values(by="SHAP Value", ascending=True)
                
                fig_shap = px.bar(
                    shap_df,
                    x="SHAP Value", y="Feature", orientation="h",
                    color="SHAP Value", color_continuous_scale="Blues",
                )
                fig_shap.update_layout(
                    showlegend=False, 
                    height=320, 
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            except Exception:
                st.info("SHAP visualization unavailable for current configuration.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Plain Language Clinical Explanation Block
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>📖 Evidence-Based Patient Explanation</div>", unsafe_allow_html=True)
        
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
        
        st.write(scientific_explanation.replace("<br/>", "\n").replace("<b>", "**").replace("</b>", "**").replace("<i>", "*").replace("</i>", "*"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # PDF Report Generator Engine
        def create_pdf():
            pdf_path = "Hypertension_Patient_Report.pdf"
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=A4,
                rightMargin=36,
                leftMargin=36,
                topMargin=36,
                bottomMargin=36
            )
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor("#1E293B"), spaceAfter=12)
            section_heading = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontSize=12, leading=15, textColor=colors.HexColor("#2563EB"), spaceBefore=10, spaceAfter=6)
            body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontSize=9.5, leading=13, textColor=colors.HexColor("#334155"))
            
            elements = []
            
            # PDF Header
            elements.append(Paragraph("🩺 Hypertension Clinical Screening Report", title_style))
            elements.append(Paragraph("<b>Automated Clinical Decision Support Summary</b>", body_style))
            elements.append(Spacer(1, 10))
            
            # Risk Summary
            elements.append(Paragraph("<b>1. Diagnostic Risk Prediction</b>", section_heading))
            elements.append(Paragraph(f"<b>Categorized Risk Profile:</b> {risk_level}", body_style))
            elements.append(Spacer(1, 8))
            
            # Probability Table
            prob_data = [
                ["Risk Category", "Low Risk", "Moderate Risk", "High Risk"],
                ["Estimated Probability", f"{probability[0]*100:.1f}%", f"{probability[1]*100:.1f}%", f"{probability[2]*100:.1f}%"]
            ]
            prob_table = Table(prob_data, colWidths=[130, 110, 110, 110])
            prob_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ]))
            elements.append(prob_table)
            elements.append(Spacer(1, 12))
            
            # Scientific Explanation Section
            elements.append(Paragraph("<b>2. Plain-Language Scientific Summary</b>", section_heading))
            elements.append(Paragraph(scientific_explanation, body_style))
            elements.append(Spacer(1, 12))
            
            # Clinical Metrics Table
            elements.append(Paragraph("<b>3. Patient Baseline Parameters</b>", section_heading))
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
                ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#F1F5F9")),
                ('BACKGROUND', (2, 0), (3, 0), colors.HexColor("#F1F5F9")),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8.5),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            elements.append(param_table)
            
            doc.build(elements)
            return pdf_path

        pdf_file = create_pdf()
        with open(pdf_file, "rb") as f:
            st.download_button("📥 Download Official Patient Medical Report (PDF)", f, file_name="Hypertension_Patient_Report.pdf")

# ================================================================
# --- PAGE 2: MODEL INTELLIGENCE ---
# ================================================================
elif page == "📊 Model Intelligence":
    st.markdown("""
        <div class='header-box'>
            <h1>📊 Machine Learning Model Intelligence</h1>
            <p>Comparative analysis and performance metrics across evaluated classifiers.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>🏆 Evaluated Models Summary</div>", unsafe_allow_html=True)
    
    results_df = pd.DataFrame({
        "Model Architecture": ["Logistic Regression", "Support Vector Machine (SVM)", "Neural Network (MLP)"],
        "Accuracy": [0.720, 0.755, 0.763],
        "Precision": [0.750, 0.759, 0.768],
        "Recall": [0.720, 0.755, 0.763],
        "F1-Score": [0.693, 0.752, 0.760],
        "ROC-AUC": [0.877, 0.880, 0.882]
    })
    
    numeric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    st.dataframe(
        results_df.style.background_gradient(cmap="Blues", subset=numeric_cols).format("{:.3f}", subset=numeric_cols),
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ================================================================
# --- PAGE 3: PROJECT OVERVIEW ---
# ================================================================
elif page == "💡 Project Overview":
    st.markdown("""
        <div class='header-box'>
            <h1>💡 About the Hypertension Screening Tool</h1>
            <p>AI-assisted health stratification designed to support early detection and clinical decision-making.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Research & Clinical Objectives
    This system applies advanced **machine learning techniques** to assess hypertension risk using non-invasive clinical indicators and lifestyle metrics.
    
    * **Target Parameters:** Age, BMI, Blood Pressure, Lipid Profile, Smoking Habits, Physical Activity, and Family History.
    * **Deployment Goal:** Enable early screening, improve public health intervention workflows, and provide evidence-based summaries for patients.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>💬 Patient & Practitioner Feedback Form</div>", unsafe_allow_html=True)
    
    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Name")
        email = st.text_input("Email (optional)")
        feedback_text = st.text_area("Feedback / Clinical Suggestions")
        submitted = st.form_submit_button("📩 Submit Feedback")

        if submitted:
            if feedback_text.strip() == "":
                st.warning("⚠️ Please enter your comments before submitting.")
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
                    st.success("✅ Feedback successfully logged. Thank you!")
                except Exception as e:
                    st.error(f"Error saving feedback: {e}")
    st.markdown("</div>", unsafe_allow_html=True)
