import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
import streamlit.components.v1 as components
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
    page_title="Hypertension Intelligence Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #f8fafc; }
        .header-box {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 2rem; border-radius: 16px; color: white; text-align: center; margin-bottom: 25px;
        }
        .header-box h1 { color: #ffffff !important; font-size: 2.2em !important; font-weight: 700 !important; margin-bottom: 6px !important; }
        .header-box p { color: #94a3b8 !important; font-size: 1em !important; margin: 0 !important; }
        .custom-card {
            background-color: #ffffff; border-radius: 14px; padding: 22px 26px;
            border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); margin-bottom: 20px;
        }
        .section-label { font-size: 1.1em; font-weight: 600; color: #334155; margin-bottom: 12px; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }
        .stButton>button {
            width: 100%; background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); color: white;
            font-weight: 600; border-radius: 10px; padding: 12px 20px; border: none;
        }
    </style>
""", unsafe_allow_html=True)

# Helper function for HTML SHAP rendering
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# --- CLINICAL RECOMMENDATIONS GENERATOR ---
def get_clinical_recommendations(risk_level, bmi, systolic, diastolic, smoking, activity, cholesterol):
    recs = []
    if risk_level in ["High Risk", "Moderate Risk"]:
        recs.append("<b>DASH Diet Adoption:</b> Emphasize vegetables, fruits, whole grains, and low-fat dairy. Limit saturated fat.")
        recs.append("<b>Sodium Reduction:</b> Restrict daily sodium intake to < 1,500 mg per day to directly attenuate arterial pressure.")
    else:
        recs.append("<b>Maintenance Diet:</b> Continue a balanced, heart-healthy dietary pattern rich in dietary fiber and antioxidants.")

    if bmi >= 25:
        recs.append("<b>Targeted Weight Reduction:</b> Aim for a 5–10% body weight reduction over 6 months to lower systematic vascular resistance.")

    if smoking == "Yes":
        recs.append("<b>Smoking Cessation Protocol:</b> Immediate cessation strongly recommended to prevent acute arterial constriction and vascular endothelial damage.")

    if activity == "Low":
        recs.append("<b>Aerobic Exercise:</b> Engage in at least 150 minutes of moderate-intensity aerobic exercise per week (e.g., brisk walking, cycling).")

    if cholesterol >= 200:
        recs.append("<b>Lipid Management:</b> Evaluate lipid profile with a clinician; consider soluble fiber supplementation and reduction of dietary trans fats.")

    return recs

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387581.png", width=65)
    st.markdown("### **Hypertension AI**")
    page = option_menu(
        menu_title=None,
        options=["🏥 Single Patient Assessment", "🧪 Scenario Analysis (What-If)", "📂 Batch Processing", "📊 Model Intelligence"],
        icons=["person-fill", "sliders", "file-earmark-spreadsheet-fill", "cpu"],
        default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "icon": {"color": "#2563eb", "font-size": "16px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "4px", "border-radius": "8px"},
            "nav-link-selected": {"background-color": "#2563eb", "color": "white"},
        },
    )

# ================================================================
# TAB 1: SINGLE PATIENT ASSESSMENT
# ================================================================
if page == "🏥 Single Patient Assessment":
    st.markdown("""
        <div class='header-box'>
            <h1>🩺 Individual Risk Assessment</h1>
            <p>Enter patient physiological parameters for single-instance stratification.</p>
        </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>📋 Patient Clinical Parameters</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 10, 100, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            occupation = st.text_input("Occupation", "Teacher")
        with col2:
            bmi = st.number_input("BMI (kg/m²)", 10.0, 60.0, 24.5)
            systolic_bp = st.number_input("Systolic BP (mmHg)", 80, 250, 120)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", 40, 150, 80)
        with col3:
            cholesterol = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 180)
            smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
            alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "High"])
            physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
            family_history = st.selectbox("Family History", ["Yes", "No"])

        predict_btn = st.button("⚡ Evaluate Risk Profile")
        st.markdown("</div>", unsafe_allow_html=True)

    if predict_btn:
        input_data = pd.DataFrame({
            'Age': [age], 'Gender': [gender], 'Occupation': [occupation], 'BMI': [bmi],
            'Systolic_BP': [systolic_bp], 'Diastolic_BP': [diastolic_bp], 'Cholesterol': [cholesterol],
            'Smoking_Status': [smoking_status], 'Alcohol_Consumption': [alcohol],
            'Physical_Activity_Level': [physical_activity], 'Family_History': [family_history]
        })

        X_new = preprocessor.transform(input_data)
        X_new_sel = X_new[:, selected_features]
        prediction = best_model.predict(X_new_sel)[0]
        probability = best_model.predict_proba(X_new_sel)[0]
        risk_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
        risk_level = risk_map[prediction]

        st.markdown(f"### Predicted Risk Level: **{risk_level}**")

        # Interactive SHAP Force Plot Feature
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>🧠 Advanced SHAP Force Plot Explanation</div>", unsafe_allow_html=True)
        try:
            explainer = shap.Explainer(best_model, X_new_sel)
            shap_values = explainer(X_new_sel)
            force_plot = shap.plots.force(explainer.expected_value[prediction], shap_values.values[0][:, prediction], feature_names=selected_features)
            st_shap(force_plot, height=200)
        except Exception:
            st.info("SHAP force plot rendering unavailable for this model configuration.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Dynamic Recommendations
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>🎯 Actionable Lifestyle & Clinical Guidelines</div>", unsafe_allow_html=True)
        recommendations = get_clinical_recommendations(risk_level, bmi, systolic_bp, diastolic_bp, smoking_status, physical_activity, cholesterol)
        for rec in recommendations:
            st.markdown(f"• {rec}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ================================================================
# TAB 2: SCENARIO ANALYSIS (WHAT-IF)
# ================================================================
elif page == "🧪 Scenario Analysis (What-If)":
    st.markdown("""
        <div class='header-box'>
            <h1>🧪 Dynamic "What-If" Scenario Tester</h1>
            <p>Adjust physiological parameters in real-time to simulate health interventions.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        sim_sys = st.slider("Systolic BP (mmHg)", 90, 200, 140)
        sim_dia = st.slider("Diastolic BP (mmHg)", 60, 120, 90)
        sim_bmi = st.slider("BMI (kg/m²)", 15.0, 45.0, 29.0)
    with c2:
        sim_chol = st.slider("Cholesterol (mg/dL)", 120, 350, 220)
        sim_smoke = st.selectbox("Simulate Smoking Cessation", ["Yes", "No"], index=0)
        sim_act = st.select_slider("Physical Activity Level", options=["Low", "Moderate", "High"], value="Low")

    sim_data = pd.DataFrame({
        'Age': [45], 'Gender': ["Male"], 'Occupation': ["Office"], 'BMI': [sim_bmi],
        'Systolic_BP': [sim_sys], 'Diastolic_BP': [sim_dia], 'Cholesterol': [sim_chol],
        'Smoking_Status': [sim_smoke], 'Alcohol_Consumption': ["Moderate"],
        'Physical_Activity_Level': [sim_act], 'Family_History': ["Yes"]
    })

    sim_X = preprocessor.transform(sim_data)[:, selected_features]
    sim_prob = best_model.predict_proba(sim_X)[0]

    st.markdown("#### Real-time Simulated Risk Distribution")
    sim_df = pd.DataFrame({
        "Category": ["Low Risk", "Moderate Risk", "High Risk"],
        "Probability": sim_prob
    })
    fig_sim = px.bar(sim_df, x="Category", y="Probability", color="Category", color_discrete_sequence=["#10b981", "#f59e0b", "#ef4444"], text_auto='.1%')
    fig_sim.update_layout(height=280, showlegend=False)
    st.plotly_chart(fig_sim, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ================================================================
# TAB 3: BATCH PROCESSING
# ================================================================
elif page == "📂 Batch Processing":
    st.markdown("""
        <div class='header-box'>
            <h1>📂 Multi-Patient Batch Processing</h1>
            <p>Upload a CSV file containing multiple patient records for bulk risk classification.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV File containing Patient Dataset", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset Preview", batch_df.head())

        if st.button("⚡ Run Batch Diagnostics"):
            try:
                batch_X = preprocessor.transform(batch_df)[:, selected_features]
                preds = best_model.predict(batch_X)
                probs = best_model.predict_proba(batch_X)

                risk_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
                batch_df["Predicted_Risk"] = [risk_map[p] for p in preds]
                batch_df["High_Risk_Probability"] = [f"{pr[2]*100:.1f}%" for pr in probs]

                st.success("Batch classification completed successfully!")
                st.dataframe(batch_df)

                csv_data = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Summary Report (CSV)", csv_data, "Hypertension_Batch_Results.csv", "text/csv")
            except Exception as e:
                st.error(f"Processing Error: Ensure CSV column headers match required input features. Details: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ================================================================
# TAB 4: MODEL INTELLIGENCE
# ================================================================
elif page == "📊 Model Intelligence":
    st.markdown("<div class='header-box'><h1>📊 Model Intelligence & Performance Metrics</h1></div>", unsafe_allow_html=True)
    st.info("Evaluation metrics across cross-validated model runs.")
