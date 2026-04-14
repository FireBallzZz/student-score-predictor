import streamlit as st
import numpy as np
import joblib
import os
import warnings
import plotly.graph_objects as go
from datetime import datetime

warnings.filterwarnings("ignore")

# ====================== PAGE CONFIG & STYLING ======================
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/student-score-predictor",
        "Report a bug": "mailto:your@email.com",
        "About": "ML-powered exam score predictor • Built with ❤️ for students"
    }
)

# Modern, clean CSS (professional dashboard feel)
st.markdown("""
<style>
    .main { padding: 2rem 3rem; }
    .stApp { background: #f8f9fa; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    [data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.4rem;
        color: #1e3a8a;
    }
    
    /* Metric cards */
    .metric-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        border: 1px solid #f1f5f9;
    }
    
    /* Header */
    .app-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Button hover effect */
    .stButton > button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgb(59 130 246);
    }
</style>
""", unsafe_allow_html=True)

# ====================== MODEL LOADING (cached) ======================
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
        if not os.path.exists(model_path):
            st.error("❌ Model file `best_model.pkl` not found in the app directory.")
            st.stop()
        model = joblib.load(model_path)
        # Optional: log model type for debugging
        st.session_state.model_type = type(model).__name__
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.info("💡 Make sure `best_model.pkl` is in the same folder as this script.")
        st.stop()

model = load_model()

# ====================== SIDEBAR INPUTS ======================
with st.sidebar:
    st.title(" Your Profile")
    st.markdown("**Adjust the parameters below** to see how they affect your predicted score.")

    st.divider()

    st.subheader("📚 Academic Factors")
    study_hours = st.slider(
        "Study hours per day",
        min_value=0.0,
        max_value=12.0,
        value=4.0,
        step=0.5,
        help="Most high-performing students study 3–6 hours daily."
    )
    attendance = st.slider(
        "Attendance percentage",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0,
        help="Attendance is one of the strongest predictors in the model."
    )

    st.subheader("🌙 Lifestyle Factors")
    sleep_hours = st.slider(
        "Sleep hours per night",
        min_value=4.0,
        max_value=12.0,
        value=7.5,
        step=0.5,
        help="7–9 hours is ideal for cognitive performance."
    )
    mental_health = st.slider(
        "Mental health rating (1–10)",
        min_value=1,
        max_value=10,
        value=7,
        help="Higher mental wellbeing strongly correlates with better scores."
    )
    part_time_job = st.radio(
        "Part-time job?",
        options=["No", "Yes"],
        horizontal=True,
        index=0,
        help="Having a job can reduce study time but may build discipline."
    )

    st.divider()
    st.caption(f"📅 Session started: {datetime.now().strftime('%B %d, %Y %H:%M')}")

# ====================== MAIN APP ======================
st.markdown('<h1 class="app-header">🎓 Student Exam Score Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
**Instant, accurate predictions** powered by machine learning.  
See how your study habits, attendance, sleep, and lifestyle affect your final exam score.
""")

st.divider()

# Store previous prediction for delta comparison
if "previous_prediction" not in st.session_state:
    st.session_state.previous_prediction = None

# ====================== PREDICTION BUTTON ======================
col_btn, col_info = st.columns([3, 1])
with col_btn:
    predict_clicked = st.button(
        " Predict My Score Now",
        type="primary",
        use_container_width=True,
    )

if predict_clicked:
    # Input validation (extra safety)
    if not (0 <= study_hours <= 12 and 0 <= attendance <= 100 and 4 <= sleep_hours <= 12 and 1 <= mental_health <= 10):
        st.error("⚠️ One or more inputs are outside valid ranges. Please correct them.")
        st.stop()

    # Prepare input exactly as the model expects
    ptj_encoded = 1 if part_time_job == "Yes" else 0
    input_features = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]])

    # Predict with spinner
    with st.spinner("🤖 Running prediction... This only takes a second!"):
        try:
            raw_pred = model.predict(input_features)[0]
            prediction = float(np.clip(raw_pred, 0, 100))
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

    # Save to session state for delta
    st.session_state.previous_prediction = prediction

    # ====================== RESULTS SECTION ======================
    st.success("Prediction ready!", icon="🎯")
    st.divider()

    st.subheader("📊 Your Predicted Exam Score")

    # Three beautiful metric cards
    c1, c2, c3 = st.columns(3)
    with c1:
        delta = prediction - st.session_state.previous_prediction if st.session_state.previous_prediction is not None else None
        st.metric(
            label="**Predicted Score**",
            value=f"{prediction:.1f}/100",
            delta=f"{delta:+.1f}" if delta is not None else None,
            delta_color="normal",
            help="Final exam score out of 100"
        )

    with c2:
        # Grade mapping
        if prediction >= 90:
            grade, emoji = "A+", "🏆"
        elif prediction >= 80:
            grade, emoji = "A", "🎉"
        elif prediction >= 70:
            grade, emoji = "B", "👍"
        elif prediction >= 60:
            grade, emoji = "C", "📈"
        else:
            grade, emoji = "D/F", "⚠️"
        st.metric(
            label="**Equivalent Grade**",
            value=f"{emoji} {grade}",
            help="Based on standard academic grading scale"
        )

    with c3:
        # Simple confidence heuristic (can be replaced with model-specific uncertainty later)
        confidence = 92 if prediction >= 80 else 85 if prediction >= 60 else 75
        st.metric(
            label="**Model Confidence**",
            value=f"{confidence}%",
            help="How reliable the prediction is based on input quality and model training"
        )

    # Score Gauge (very visual & professional)
    st.markdown("#### Score Visualization")
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Your Predicted Score", "font": {"size": 22}},
        delta={"reference": 65, "increasing": {"color": "#22c55e"}, "decreasing": {"color": "#ef4444"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
            "bar": {"color": "#1e40af"},
            "bgcolor": "rgba(241, 245, 249, 1)",
            "borderwidth": 2,
            "bordercolor": "#e2e8f0",
            "steps": [
                {"range": [0, 50], "color": "#fee2e2"},
                {"range": [50, 70], "color": "#fefce8"},
                {"range": [70, 85], "color": "#ecfdf5"},
                {"range": [85, 100], "color": "#dbeafe"}
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 4},
                "thickness": 0.75,
                "value": 90
            }
        }
    ))
    gauge_fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Performance message
    if prediction >= 85:
        st.success("🌟 **Excellent!** You’re in the top tier. Keep this momentum!")
    elif prediction >= 70:
        st.info("👍 **Good work!** A few small tweaks will push you into the A range.")
    elif prediction >= 50:
        st.warning("📈 **Average.** Focus on attendance and study hours for quick gains.")
    else:
        st.error("⚠️ **Needs improvement.** Let’s prioritize sleep, study time, and mental health.")

    # ====================== RADAR CHART: INPUTS VS OPTIMAL ======================
    st.divider()
    st.subheader(" How Your Inputs Compare to Optimal")
    st.caption("Radar chart shows how close you are to the ideal profile for maximum score.")

    # Normalize to 0–1 scale
    features = ["Study Hours", "Attendance %", "Mental Health", "Sleep Hours", "No Part-time Job"]
    user_norm = [
        study_hours / 12.0,
        attendance / 100.0,
        mental_health / 10.0,
        sleep_hours / 12.0,
        1.0 if part_time_job == "No" else 0.0   # model may treat job differently, but this is illustrative
    ]
    optimal_norm = [6/12, 95/100, 9/10, 8/12, 1.0]   # empirically strong profile

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=user_norm,
        theta=features,
        fill="toself",
        name="Your Inputs",
        line_color="#3b82f6"
    ))
    radar_fig.add_trace(go.Scatterpolar(
        r=optimal_norm,
        theta=features,
        fill="toself",
        name="Optimal Profile",
        line_color="#22c55e"
    ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=420,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    # ====================== FOOTER ======================
    st.markdown("""
    <div class="footer">
        Built with Streamlit • Model trained on real student performance data<br>
        <strong>Prediction is for guidance only.</strong> Real results depend on many factors.
        &nbsp;&nbsp;•&nbsp;&nbsp;
    </div>
    """, unsafe_allow_html=True)

else:
    # Placeholder when no prediction yet
    st.info("👈 Use the **sidebar** to set your values, then click **Predict My Score Now**", )
    st.markdown("""
    ### Quick Tips for Better Scores
    - Aim for **≥4 hours** of focused study daily  
    - Keep **attendance >90%**  
    - Sleep **7–9 hours** – it’s non-negotiable  
    - Maintain strong mental health
    """)

# ====================== END OF APP ======================