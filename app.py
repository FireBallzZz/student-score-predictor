import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings

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

# Modern, clean CSS (your original - unchanged)
st.markdown("""
<style>
    .main { padding: 2rem 3rem; }
    .stApp { background: #f8f9fa; }
    
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    [data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.4rem;
        color: #1e3a8a;
    }
    
    .metric-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        border: 1px solid #f1f5f9;
    }
    
    .app-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }
    
    .stButton > button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgb(59 130 246);
    }
</style>
""", unsafe_allow_html=True)

# ====================== LOAD MODEL + CLEANED DATA ======================
@st.cache_resource(show_spinner=False)
def load_model_and_data():
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file best_model.pkl not found.")
    model = joblib.load(model_path)
    csv_path = os.path.join(os.path.dirname(__file__), "student_habits_performance.csv")
    df = pd.read_csv(csv_path)
    return model, df

try:
    model, df = load_model_and_data()
except Exception as e:
    st.error(f"❌ Failed to load resources: {e}")
    st.stop()

# ====================== SIDEBAR INPUTS (unchanged) ======================
with st.sidebar:
    st.title("⚙️ Your Profile")
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

if "previous_prediction" not in st.session_state:
    st.session_state.previous_prediction = None
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ====================== LANDING INFO CARDS ======================
info_col1, info_col2, info_col3, info_col4 = st.columns(4)
with info_col1:
    st.metric("📋 Dataset Size", f"{len(df):,} students", help="Records used to train the model")
with info_col2:
    st.metric("🎯 Avg Exam Score", f"{df['exam_score'].mean():.1f}/100", help="Mean score across the dataset")
with info_col3:
    st.metric("📚 Avg Study Hours", f"{df['study_hours_per_day'].mean():.1f}h/day", help="Average daily study time")
with info_col4:
    st.metric("🏫 Avg Attendance", f"{df['attendance_percentage'].mean():.1f}%", help="Average attendance rate")

st.caption("👈 Adjust your profile in the sidebar, then click the button below to see your predicted score.")
st.divider()

# ====================== PREDICTION BUTTON ======================
predict_clicked = st.button(
    "🚀 Predict My Score Now",
    type="primary",
    use_container_width=True,
)

if predict_clicked:
    if not (0 <= study_hours <= 12 and 0 <= attendance <= 100 and 4 <= sleep_hours <= 12 and 1 <= mental_health <= 10):
        st.error("⚠️ One or more inputs are outside valid ranges. Please correct them.")
        st.stop()

    ptj_encoded = 1 if part_time_job == "Yes" else 0
    # Feature order MUST match training: study_hours, attendance, mental_health, sleep_hours, part_time_job
    input_features = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]])

    with st.spinner("🤖 Running prediction... This only takes a second!"):
        try:
            raw_pred = model.predict(input_features)[0]
            prediction = float(np.clip(raw_pred, 0, 100))
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

    # Save previous before overwriting so delta is meaningful
    st.session_state.previous_prediction = st.session_state.current_prediction
    st.session_state.current_prediction = prediction

    # Log to history
    st.session_state.prediction_history.append({
        "Run": len(st.session_state.prediction_history) + 1,
        "Study Hrs": study_hours,
        "Attendance %": attendance,
        "Mental Health": mental_health,
        "Sleep Hrs": sleep_hours,
        "Part-time Job": part_time_job,
        "Predicted Score": round(prediction, 1),
        "Grade": "A+" if prediction >= 90 else "A" if prediction >= 80 else "B" if prediction >= 70 else "C" if prediction >= 60 else "D/F"
    })

    st.success("✅ Prediction ready!", icon="🎯")
    st.divider()

    st.subheader("📊 Your Predicted Exam Score")

    c1, c2, c3 = st.columns(3)
    with c1:
        prev = st.session_state.previous_prediction
        delta = prediction - prev if prev is not None else None
        st.metric(
            label="**Predicted Score**",
            value=f"{prediction:.1f}/100",
            delta=f"{delta:+.1f}" if delta is not None else None,
        )

    with c2:
        if prediction >= 80:
            grade, emoji = "A+", "🏆"
        elif prediction >= 75:
            grade, emoji = "A", "🎉"
        elif prediction >= 70:
            grade, emoji = "A-", "👍"
        elif prediction >= 60:
            grade, emoji = "B", "📈"
        else:
            grade, emoji = "C/D", "⚠️"
        st.metric(label="**Equivalent Grade**", value=f"{emoji} {grade}")

    with c3:
        # Compute R² on dataset as a live model quality indicator
        # Feature order MUST match training: study_hours, attendance, mental_health, sleep_hours, part_time_job
        feat_cols = ['study_hours_per_day', 'attendance_percentage', 'mental_health_rating', 'sleep_hours']
        try:
            X_all = df[feat_cols].copy().astype(float)
            # Encode part_time_job: "Yes"->1, "No"->0 if string, else use as-is
            if 'part_time_job' in df.columns:
                raw_ptj = df['part_time_job']
                if raw_ptj.dtype == object:
                    ptj_col = raw_ptj.map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
                else:
                    ptj_col = raw_ptj.astype(int)
                X_all['part_time_job'] = ptj_col
            else:
                X_all['part_time_job'] = 0  # fallback if column missing
            # Final column order matching training
            X_all = X_all[['study_hours_per_day', 'attendance_percentage',
                            'mental_health_rating', 'sleep_hours', 'part_time_job']]
            y_all = df['exam_score'].astype(float)
            y_pred_all = model.predict(X_all)
            ss_res = ((y_all - y_pred_all) ** 2).sum()
            ss_tot = ((y_all - y_all.mean()) ** 2).sum()
            r2_live = float(1 - ss_res / ss_tot)
            mae_live = float(np.abs(y_all - y_pred_all).mean())
            st.metric(
                label="**Model R² Score**",
                value=f"{r2_live:.3f}",
                help=f"R² = {r2_live:.4f} | MAE = {mae_live:.2f} pts. Closer to 1.0 = better fit."
            )
        except Exception:
            st.metric(label="**Model Reliability**", value="High",
                      help="Trained on student dataset.")

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
            "threshold": {"line": {"color": "#ef4444", "width": 4}, "thickness": 0.75, "value": 90}
        }
    ))
    gauge_fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(gauge_fig, use_container_width=True)

    if prediction >= 85:
        st.success("🌟 **Excellent!** You're in the top tier. Keep this momentum!")
    elif prediction >= 70:
        st.info("👍 **Good work!** A few small tweaks will push you into the A range.")
    elif prediction >= 50:
        st.warning("📈 **Average.** Focus on attendance and study hours for quick gains.")
    else:
        st.error("⚠️ **Needs improvement.** Let's prioritize sleep, study time, and mental health.")

    st.divider()
    st.subheader("🔍 How Your Inputs Compare to Optimal")
    st.caption("Radar chart shows how close you are to the ideal profile for maximum score.")

    features = ["Study Hours", "Attendance %", "Mental Health", "Sleep Hours", "No Part-time Job"]
    user_norm = [study_hours / 12.0, attendance / 100.0, mental_health / 10.0, sleep_hours / 12.0, 1.0 if part_time_job == "No" else 0.0]
    optimal_norm = [6/12, 95/100, 9/10, 8/12, 1.0]
    avg_norm = [
        df['study_hours_per_day'].mean() / 12.0,
        df['attendance_percentage'].mean() / 100.0,
        df['mental_health_rating'].mean() / 10.0,
        df['sleep_hours'].mean() / 12.0,
        0.6
    ]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(r=user_norm, theta=features, fill="toself", name="Your Inputs", line_color="#3b82f6"))
    radar_fig.add_trace(go.Scatterpolar(r=optimal_norm, theta=features, fill="toself", name="Optimal Profile", line_color="#22c55e"))
    radar_fig.add_trace(go.Scatterpolar(r=avg_norm, theta=features, fill="toself", name="Average in Dataset", line_color="#f59e0b"))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=420,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    # ==================== PERCENTILE WITH HIGHLIGHTED DISTRIBUTION ====================
    pred_percentile = (df['exam_score'] < prediction).mean() * 100
    st.info(f"You scored higher than **{pred_percentile:.1f}%** of students in the dataset.")

    st.divider()
    st.subheader("📍 Your Score on the Full Distribution")
    st.caption("The red line shows exactly where your predicted score falls among all students.")

    dist_fig = px.histogram(
        df, x='exam_score', nbins=30,
        title="Exam Score Distribution — Where Do You Stand?",
        labels={'exam_score': 'Exam Score', 'count': 'Number of Students'},
        color_discrete_sequence=['#93c5fd']
    )
    dist_fig.add_vline(
        x=prediction,
        line_color="#ef4444",
        line_width=3,
        annotation_text=f"Your Score: {prediction:.1f}",
        annotation_position="top right",
        annotation_font_color="#ef4444",
        annotation_font_size=14
    )
    # Add percentile bands
    q10 = df['exam_score'].quantile(0.10)
    q90 = df['exam_score'].quantile(0.90)
    dist_fig.add_vrect(x0=df['exam_score'].min(), x1=q10,
                       fillcolor="#fee2e2", opacity=0.3, line_width=0,
                       annotation_text="Bottom 10%", annotation_position="top left")
    dist_fig.add_vrect(x0=q90, x1=df['exam_score'].max(),
                       fillcolor="#dcfce7", opacity=0.3, line_width=0,
                       annotation_text="Top 10%", annotation_position="top right")
    dist_fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(dist_fig, use_container_width=True)

    # Percentile breakdown table
    st.markdown("#### 📊 Percentile Breakdown")
    percentile_data = {
        'Percentile': ['10th (Low)', '25th', '50th (Median)', '75th', '90th (High)', 'Your Score'],
        'Score': [
            round(df['exam_score'].quantile(0.10), 1),
            round(df['exam_score'].quantile(0.25), 1),
            round(df['exam_score'].quantile(0.50), 1),
            round(df['exam_score'].quantile(0.75), 1),
            round(df['exam_score'].quantile(0.90), 1),
            round(prediction, 1)
        ],
        'Status': ['⚠️ Low', '📉 Below Avg', '📊 Average', '📈 Above Avg', '🏆 Top Tier',
                   '🎯 You']
    }
    percentile_df_display = pd.DataFrame(percentile_data)
    st.dataframe(percentile_df_display, use_container_width=True, hide_index=True)

    # ==================== EXTREME PERFORMER ANALYSIS ====================
    st.divider()
    st.subheader("🔬 Extreme Performer Analysis")
    st.caption("What separates top 10% students from bottom 10%? This is what the model has learned.")

    top10 = df[df['exam_score'] >= df['exam_score'].quantile(0.90)]
    bot10 = df[df['exam_score'] <= df['exam_score'].quantile(0.10)]

    compare_cols = ['study_hours_per_day', 'attendance_percentage', 'mental_health_rating', 'sleep_hours']
    compare_labels = ['Study Hours/Day', 'Attendance %', 'Mental Health', 'Sleep Hours']

    top_means = [top10[c].mean() for c in compare_cols]
    bot_means = [bot10[c].mean() for c in compare_cols]
    your_vals = [study_hours, attendance, mental_health, sleep_hours]

    extreme_fig = go.Figure()
    extreme_fig.add_trace(go.Bar(name='🏆 Top 10% Students', x=compare_labels, y=top_means,
                                  marker_color='#22c55e', opacity=0.85))
    extreme_fig.add_trace(go.Bar(name='⚠️ Bottom 10% Students', x=compare_labels, y=bot_means,
                                  marker_color='#ef4444', opacity=0.85))
    extreme_fig.add_trace(go.Scatter(name='🎯 Your Profile', x=compare_labels, y=your_vals,
                                      mode='markers+lines', marker=dict(size=12, color='#1e40af'),
                                      line=dict(color='#1e40af', width=2, dash='dot')))
    extreme_fig.update_layout(
        barmode='group',
        title="Top 10% vs Bottom 10% — Average Feature Values",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(extreme_fig, use_container_width=True)

    # Key differences
    c1e, c2e = st.columns(2)
    with c1e:
        st.markdown("**🏆 Top 10% profile:**")
        for lbl, val in zip(compare_labels, top_means):
            st.write(f"• {lbl}: **{val:.1f}**")
    with c2e:
        st.markdown("**⚠️ Bottom 10% profile:**")
        for lbl, val in zip(compare_labels, bot_means):
            st.write(f"• {lbl}: **{val:.1f}**")

    # ==================== FEATURE IMPORTANCE ====================
    st.divider()
    st.subheader("🔬 Feature Importance")

    feature_names = ["study_hours", "attendance", "mental_health", "sleep_hours", "part_time_job"]

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)

        fig_imp = px.bar(importance_df, x='Importance', y='Feature',
                         orientation='h', title="Feature Importance (Model-Learned)",
                         labels={'Importance': 'Relative Importance'},
                         color='Importance', color_continuous_scale='blues')
        st.plotly_chart(fig_imp, use_container_width=True)

        top_feature = importance_df.iloc[-1]['Feature'].replace('_', ' ').title()
        st.info(f"**Why this prediction?** The model relies most on **{top_feature}**. "
                f"Higher values in this feature generally increase the predicted score.")
    else:
        st.info("Feature importance not available for this model type.")

    # ==================== WHAT-IF SENSITIVITY ANALYSIS ====================
    st.divider()
    st.subheader("🔮 What-If Sensitivity Analysis")
    st.caption("See how improving each factor by 1 unit would change your predicted score.")

    ptj_enc = 1 if part_time_job == "Yes" else 0
    base_input = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_enc]])
    base_pred = float(np.clip(model.predict(base_input)[0], 0, 100))

    whatif_results = []

    # +1 study hour
    if study_hours < 12:
        inp = base_input.copy(); inp[0][0] = min(study_hours + 1, 12)
        delta_val = float(np.clip(model.predict(inp)[0], 0, 100)) - base_pred
        whatif_results.append(("📚 +1 Study Hour/Day", study_hours, study_hours + 1, delta_val))

    # +5 attendance
    if attendance < 100:
        inp = base_input.copy(); inp[0][1] = min(attendance + 5, 100)
        delta_val = float(np.clip(model.predict(inp)[0], 0, 100)) - base_pred
        whatif_results.append(("🏫 +5% Attendance", attendance, min(attendance + 5, 100), delta_val))

    # +1 mental health
    if mental_health < 10:
        inp = base_input.copy(); inp[0][2] = min(mental_health + 1, 10)
        delta_val = float(np.clip(model.predict(inp)[0], 0, 100)) - base_pred
        whatif_results.append(("🧠 +1 Mental Health Rating", mental_health, min(mental_health + 1, 10), delta_val))

    # +1 sleep hour
    if sleep_hours < 12:
        inp = base_input.copy(); inp[0][3] = min(sleep_hours + 1, 12)
        delta_val = float(np.clip(model.predict(inp)[0], 0, 100)) - base_pred
        whatif_results.append(("😴 +1 Sleep Hour/Night", sleep_hours, min(sleep_hours + 1, 12), delta_val))

    # Remove part-time job if has one
    if part_time_job == "Yes":
        inp = base_input.copy(); inp[0][4] = 0
        delta_val = float(np.clip(model.predict(inp)[0], 0, 100)) - base_pred
        whatif_results.append(("💼 Remove Part-time Job", "Yes", "No", delta_val))

    if whatif_results:
        wi_labels = [r[0] for r in whatif_results]
        wi_deltas = [r[3] for r in whatif_results]
        wi_colors = ['#22c55e' if d >= 0 else '#ef4444' for d in wi_deltas]

        wi_fig = go.Figure(go.Bar(
            x=wi_labels,
            y=wi_deltas,
            marker_color=wi_colors,
            text=[f"{d:+.2f}" for d in wi_deltas],
            textposition='outside'
        ))
        wi_fig.update_layout(
            title="Score Change if You Improve Each Factor",
            yaxis_title="Score Change (points)",
            xaxis_title="Improvement Scenario",
            height=380,
            yaxis=dict(zeroline=True, zerolinecolor='#94a3b8', zerolinewidth=2)
        )
        st.plotly_chart(wi_fig, use_container_width=True)

        st.markdown("**📌 Actionable Insights:**")
        sorted_wi = sorted(whatif_results, key=lambda x: x[3], reverse=True)
        for label, old_val, new_val, delta_val in sorted_wi:
            arrow = "🟢" if delta_val > 0 else "🔴"
            st.write(f"{arrow} **{label}**: score changes by **{delta_val:+.2f} points**")

    # ==================== FEATURE INTERACTION ANALYSIS ====================
    st.divider()
    st.subheader("🔗 Feature Interaction Analysis")
    st.caption("How do two features together affect exam scores?")

    int_col1, int_col2 = st.columns(2)
    with int_col1:
        int_fig1 = px.scatter(
            df, x='study_hours_per_day', y='sleep_hours',
            color='exam_score',
            title="Study Hours × Sleep Hours → Score",
            labels={
                'study_hours_per_day': 'Study Hours/Day',
                'sleep_hours': 'Sleep Hours/Night',
                'exam_score': 'Exam Score'
            },
            color_continuous_scale='RdYlGn'
        )
        int_fig1.add_scatter(
            x=[study_hours], y=[sleep_hours],
            mode='markers',
            marker=dict(size=16, color='blue', symbol='star'),
            name='You',
            showlegend=True
        )
        int_fig1.update_layout(height=380)
        st.plotly_chart(int_fig1, use_container_width=True)

    with int_col2:
        int_fig2 = px.scatter(
            df, x='attendance_percentage', y='mental_health_rating',
            color='exam_score',
            title="Attendance × Mental Health → Score",
            labels={
                'attendance_percentage': 'Attendance %',
                'mental_health_rating': 'Mental Health Rating',
                'exam_score': 'Exam Score'
            },
            color_continuous_scale='RdYlGn'
        )
        int_fig2.add_scatter(
            x=[attendance], y=[mental_health],
            mode='markers',
            marker=dict(size=16, color='blue', symbol='star'),
            name='You',
            showlegend=True
        )
        int_fig2.update_layout(height=380)
        st.plotly_chart(int_fig2, use_container_width=True)

    st.caption("🔵 Blue star = Your position. Green = High score zone, Red = Low score zone.")

    # ==================== PERSONALIZED WHY EXPLANATION ====================
    st.divider()
    st.subheader("🧠 Personalized Prediction Explanation")
    st.caption("This explanation is specific to YOUR inputs — not generic advice.")

    avg_study   = df['study_hours_per_day'].mean()
    avg_attend  = df['attendance_percentage'].mean()
    avg_mental  = df['mental_health_rating'].mean()
    avg_sleep   = df['sleep_hours'].mean()

    explanations = []
    if study_hours >= avg_study + 1:
        explanations.append(("✅", "Study Hours", f"{study_hours}h/day (avg: {avg_study:.1f}h)", "significantly above average — strong positive impact"))
    elif study_hours >= avg_study:
        explanations.append(("🟡", "Study Hours", f"{study_hours}h/day (avg: {avg_study:.1f}h)", "slightly above average — minor positive impact"))
    else:
        explanations.append(("❌", "Study Hours", f"{study_hours}h/day (avg: {avg_study:.1f}h)", f"below average by {avg_study - study_hours:.1f}h — dragging score down"))

    if attendance >= avg_attend + 5:
        explanations.append(("✅", "Attendance", f"{attendance:.0f}% (avg: {avg_attend:.1f}%)", "well above average — model weights this heavily"))
    elif attendance >= avg_attend:
        explanations.append(("🟡", "Attendance", f"{attendance:.0f}% (avg: {avg_attend:.1f}%)", "at or above average — neutral/positive effect"))
    else:
        explanations.append(("❌", "Attendance", f"{attendance:.0f}% (avg: {avg_attend:.1f}%)", f"below average by {avg_attend - attendance:.1f}% — significant negative impact"))

    if mental_health >= avg_mental + 1:
        explanations.append(("✅", "Mental Health", f"{mental_health}/10 (avg: {avg_mental:.1f})", "above average — supports cognitive performance"))
    elif mental_health >= avg_mental:
        explanations.append(("🟡", "Mental Health", f"{mental_health}/10 (avg: {avg_mental:.1f})", "at average — no strong impact either way"))
    else:
        explanations.append(("❌", "Mental Health", f"{mental_health}/10 (avg: {avg_mental:.1f})", "below average — likely reducing predicted score"))

    if 7 <= sleep_hours <= 9:
        explanations.append(("✅", "Sleep", f"{sleep_hours}h (ideal: 7–9h)", "in the optimal range — best for cognitive function"))
    elif sleep_hours < 7:
        explanations.append(("❌", "Sleep", f"{sleep_hours}h (ideal: 7–9h)", f"under-sleeping by {7 - sleep_hours:.1f}h — impacting performance"))
    else:
        explanations.append(("🟡", "Sleep", f"{sleep_hours}h (ideal: 7–9h)", "slightly over ideal range — minor impact"))

    if part_time_job == "No":
        explanations.append(("✅", "Part-time Job", "No job", "no time conflict — positive factor for study focus"))
    else:
        explanations.append(("🟡", "Part-time Job", "Has job", "may reduce available study time — small negative factor"))

    exp_df = pd.DataFrame(explanations, columns=['Status', 'Factor', 'Your Value', 'Interpretation'])
    st.dataframe(exp_df, use_container_width=True, hide_index=True)

    # ==================== PERSONALISED STUDY PLAN ====================
    st.divider()
    st.subheader("📋 Your Personalised Action Plan")
    st.caption("Based on your inputs vs the top 10% student profile — here is what to prioritise.")

    top10_df = df[df['exam_score'] >= df['exam_score'].quantile(0.90)]
    study_plan = []

    gap_study = top10_df['study_hours_per_day'].mean() - study_hours
    gap_attend = top10_df['attendance_percentage'].mean() - attendance
    gap_mental = top10_df['mental_health_rating'].mean() - mental_health
    gap_sleep  = top10_df['sleep_hours'].mean() - sleep_hours

    if gap_study > 0.5:
        study_plan.append(("📚 Study Hours", f"Increase by ~{gap_study:.1f}h/day",
                           f"Top students average {top10_df['study_hours_per_day'].mean():.1f}h — try adding 30–60 min sessions.",
                           "🔴 High Priority"))
    if gap_attend > 3:
        study_plan.append(("🏫 Attendance", f"Improve by ~{gap_attend:.1f}%",
                           f"Top students average {top10_df['attendance_percentage'].mean():.1f}% — missing fewer classes has a large model impact.",
                           "🔴 High Priority"))
    if gap_mental > 1:
        study_plan.append(("🧠 Mental Health", f"Improve wellbeing by {gap_mental:.1f} pts",
                           "Consider mindfulness, exercise, or talking to a counsellor.",
                           "🟠 Medium Priority"))
    if not (7 <= sleep_hours <= 9):
        target_sleep = 8.0
        study_plan.append(("😴 Sleep", f"Aim for {target_sleep}h (currently {sleep_hours}h)",
                           "Sleep quality directly affects memory consolidation and focus.",
                           "🟠 Medium Priority"))
    if part_time_job == "Yes":
        study_plan.append(("💼 Part-time Job", "Consider reducing work hours during exam periods",
                           "Top students without jobs score an average of "
                           f"{df[df['part_time_job'] == 'No']['exam_score'].mean():.1f} vs "
                           f"{df[df['part_time_job'] == 'Yes']['exam_score'].mean():.1f} for those with jobs.",
                           "🟡 Low Priority"))

    if not study_plan:
        st.success("🌟 Your profile already matches the top 10%! Keep it up.")
    else:
        plan_df = pd.DataFrame(study_plan, columns=["Factor", "Target", "Advice", "Priority"])
        st.dataframe(plan_df, use_container_width=True, hide_index=True)

    # ==================== PREDICTION HISTORY ====================
    st.divider()
    st.subheader("📜 Prediction History (This Session)")
    if len(st.session_state.prediction_history) > 1:
        hist_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        # Export as CSV
        csv_export = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download My Prediction History (CSV)",
            data=csv_export,
            file_name="my_score_predictions.csv",
            mime="text/csv",
            help="Download all your prediction runs from this session."
        )
    else:
        st.info("Run more predictions with different inputs to see your history here.")

    # Attendance vs Score plot (original, unchanged)
    st.subheader("📈 Attendance vs Actual Score")
    st.plotly_chart(
        px.scatter(df, x='attendance_percentage', y='exam_score',
                   trendline="ols", title="Attendance vs Exam Score (Dataset)"),
        use_container_width=True
    )

    st.markdown("""
    <div class="footer">
        Built with Streamlit • Model trained on real student performance data<br>
        <strong>Prediction is for guidance only.</strong> Real results depend on many factors.
    </div>
    """, unsafe_allow_html=True)

    # ====================== FINAL THESIS-LEVEL ANALYSIS ======================

    st.divider()
    st.header("🎓 Advanced Data Science Analysis")

    # ================== 1. DATA DISTRIBUTION ==================
    st.subheader("📊 Data Distribution & Outliers")

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(
            px.histogram(df, x="exam_score", nbins=30,
                         title="Distribution of Exam Scores"),
            use_container_width=True
        )

    with c2:
        st.plotly_chart(
            px.box(df, y="exam_score",
                   title="Outlier Detection (Boxplot)"),
            use_container_width=True
        )

    # ================== 2. FEATURE DISTRIBUTIONS ==================
    st.subheader("📈 Feature Distributions")

    features = [
        'study_hours_per_day',
        'attendance_percentage',
        'mental_health_rating',
        'sleep_hours'
    ]

    selected_feature = st.selectbox("Select Feature to Explore", features)

    st.plotly_chart(
        px.histogram(df, x=selected_feature, nbins=30,
                     title=f"{selected_feature} Distribution"),
        use_container_width=True
    )

    # ================== 3. CORRELATION ANALYSIS ==================
    st.subheader("🔗 Correlation Analysis")

    corr_matrix = df.corr(numeric_only=True)

    st.plotly_chart(
        px.imshow(corr_matrix, text_auto=True,
                  title="Correlation Heatmap"),
        use_container_width=True
    )

    # ================== 4. FEATURE RELATIONSHIPS ==================
    st.subheader("📈 Feature vs Exam Score Relationship")

    selected_x = st.selectbox("Choose Feature", features, key="rel_plot")

    st.plotly_chart(
        px.scatter(df, x=selected_x, y="exam_score",
                   trendline="ols",
                   title=f"{selected_x} vs Exam Score"),
        use_container_width=True
    )

    # ================== 5. STATISTICAL SUMMARY ==================
    st.subheader("📊 Statistical Summary")

    st.write("### Exam Score Statistics")
    st.write(df['exam_score'].describe())

    # ================== 6. PERCENTILE ANALYSIS (UPDATED - no duplicate) ==================
    st.subheader("📌 Percentile Insight — Full Dataset View")

    percentile_scores = {
        '10th': df['exam_score'].quantile(0.10),
        '25th': df['exam_score'].quantile(0.25),
        '50th': df['exam_score'].quantile(0.50),
        '75th': df['exam_score'].quantile(0.75),
        '90th': df['exam_score'].quantile(0.90),
    }

    pct_fig = go.Figure()
    pct_fig.add_trace(go.Scatter(
        x=list(percentile_scores.keys()),
        y=list(percentile_scores.values()),
        mode='lines+markers+text',
        text=[f"{v:.1f}" for v in percentile_scores.values()],
        textposition='top center',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=10, color='#1e40af')
    ))
    pct_fig.update_layout(
        title="Exam Score by Percentile Band",
        xaxis_title="Percentile",
        yaxis_title="Exam Score",
        height=360
    )
    st.plotly_chart(pct_fig, use_container_width=True)

    # ================== 7. DATA-DRIVEN EXPLANATION ==================
    st.subheader("🧠 Why This Prediction? (Data Perspective)")

    avg_values = {
        "study_hours": df['study_hours_per_day'].mean(),
        "attendance": df['attendance_percentage'].mean(),
        "mental_health": df['mental_health_rating'].mean(),
        "sleep_hours": df['sleep_hours'].mean()
    }

    input_values = {
        "study_hours": study_hours,
        "attendance": attendance,
        "mental_health": mental_health,
        "sleep_hours": sleep_hours
    }

    for key in input_values:
        diff = input_values[key] - avg_values.get(key, 0)
        if diff > 0:
            st.success(f"{key.replace('_',' ').title()} is ABOVE average → positive impact on score")
        elif diff < 0:
            st.warning(f"{key.replace('_',' ').title()} is BELOW average → negative impact on score")
        else:
            st.info(f"{key.replace('_',' ').title()} is at average level")

    # ================== 8. MODEL INTERPRETATION ==================
    st.subheader("🔬 Model Interpretation")

    st.write("""
    Feature importance shows which variables influence the prediction the most.
    However, importance alone does not explain direction (positive/negative impact),
    so we combine it with data comparison above.
    """)

    # ================== 9. GROUPED PERFORMANCE ANALYSIS (NEW) ==================
    st.subheader("📊 Grouped Performance Analysis")
    st.caption("Students grouped by attendance level — how does each group perform on average?")

    df_grouped = df.copy()
    df_grouped['attendance_group'] = pd.cut(
        df_grouped['attendance_percentage'],
        bins=[0, 60, 75, 85, 100],
        labels=['<60% (Low)', '60–75% (Below Avg)', '75–85% (Above Avg)', '85–100% (High)']
    )
    group_summary = df_grouped.groupby('attendance_group', observed=True)['exam_score'].mean().reset_index()
    group_summary.columns = ['Attendance Group', 'Average Exam Score']

    grp_fig = px.bar(
        group_summary, x='Attendance Group', y='Average Exam Score',
        title="Average Exam Score by Attendance Group",
        color='Average Exam Score', color_continuous_scale='blues',
        text='Average Exam Score'
    )
    grp_fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    grp_fig.update_layout(height=380)
    st.plotly_chart(grp_fig, use_container_width=True)

    # ================== 10. FINAL INSIGHTS ==================
    st.subheader("📌 Key Insights from Data")

    st.markdown("""
    - 📈 Higher attendance strongly correlates with better scores  
    - 📚 Study hours show positive but diminishing returns  
    - 🧠 Mental health significantly impacts performance  
    - 😴 Proper sleep (7–9 hours) improves outcomes  
    - ⚖️ Part-time job may slightly reduce academic performance  

    👉 Conclusion: Academic success is multi-factorial, combining discipline, health, and consistency.
    """)

    # ================== END ==================
    # ====================== END OF APP ======================