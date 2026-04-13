import streamlit as st
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Student Score Predictor", page_icon="🎓", layout="centered")

st.markdown("""
<style>
    .main { padding: 2rem; }
    .stSlider > div > div { padding-top: 0.3rem; }
    .metric-row { display: flex; gap: 12px; margin-bottom: 1.5rem; }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model
try:
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# Header
st.markdown("## 🎓 Student Exam Score Predictor")
st.markdown("Adjust the inputs below to predict your expected exam score.")
st.divider()

# Live summary metrics
col1, col2, col3 = st.columns(3)

# Academic inputs
st.markdown("### 📚 Academic inputs")
study_hours = st.slider("Study hours per day", 0.0, 12.0, 2.0, step=0.5)
attendance  = st.slider("Attendance percentage", 0.0, 100.0, 80.0, step=1.0)

# Lifestyle inputs
st.markdown("### 🌙 Lifestyle inputs")
sleep_hours   = st.slider("Sleep hours per night", 0.0, 12.0, 7.0, step=0.5)
mental_health = st.slider("Mental health rating (1–10)", 1, 10, 5)
part_time_job = st.radio("Part-time job", ["No", "Yes"], horizontal=True)

# Live metric summary
col1.metric("Study hours", f"{study_hours:.1f} hrs")
col2.metric("Attendance",  f"{attendance:.0f}%")
col3.metric("Sleep",       f"{sleep_hours:.1f} hrs")

st.divider()

# Predict button
if st.button("🔍 Predict exam score", use_container_width=True, type="primary"):
    ptj_encoded = 1 if part_time_job == "Yes" else 0
    input_data  = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]])
    prediction  = model.predict(input_data)[0]
    prediction  = max(0, min(100, prediction))

    # Result display
    st.markdown("---")
    st.markdown("### 🎯 Prediction result")

    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric(label="Predicted score", value=f"{prediction:.1f} / 100")

    with r2:
        if prediction >= 85:
            st.success("🌟 Excellent — keep up the great work!")
        elif prediction >= 70:
            st.info("👍 Good — a little more effort will push you higher.")
        elif prediction >= 50:
            st.warning("📈 Average — focus on attendance and study hours.")
        else:
            st.error("⚠️ Needs improvement — consider reducing distractions.")

    st.progress(int(prediction) / 100)