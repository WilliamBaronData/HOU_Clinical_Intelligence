import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page settings
st.set_set_page_config(page_title="Houston Clinical Analytics", layout="wide", page_icon="🏥")

# Main Title
st.title("🏥 Houston Methodist - Clinical Risk AI")
st.markdown("### Prediction of Infection Risk (CLASBSI) with 98.24% of accuracy")

# 1. Load of data
@st.cache_data
def load_data():
    # Assuming your output file is named like this in the data folder
    return pd.read_csv('data/clinical_prediction_forbi.csv')

df =load_data()

# 2. Key Metrics (KPIs)
col1, col2, col3, col4, = st.columns(4)
with col1:
    st.metric("Total Patients", len(df))
with col2:
    high_risk = len(df[df['risk_probability'] > 0.7])
    st.metric("High-Risk Cases", high_risk, delta=f"{high_risk/len(df):.1%}", delta_color="inverse")
with col3:
    avg_risk = df['risk_probablity']. mean()
    st.metric("Average Risk", f"{avg_risk: .2%}")
with col4:
    st.metric("Model Accuracy", "98.24%")

st.divider()

# 3. Interactive Graphics
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Risk per Care Unit")
    fig_unit = px.sunburst(df, path=['care_unit', 'severity_level'], values='risk_probability',
                        color='risk_probability', color_continuos_scale='Reds')
    st.ploty_chart(fig_unit, use_container_width=True)

with col_right:
    st.subheader("Probability Distribution of Infection")
    fig_hist = px.histogram(df, x="risk_probability", nbins=20, color="care_unit",
                        marginal="box", title="Dispersion Analysis")
    st.ploty_chart(fig_hist, use_container_width=True)

# 4. Data Table for Medical Personnel
st.subheader("📋 Priority Intervention List")
# Highlight those at high risk in red
st,dataframe(df.style.background_gradient(subset=['risk_probability'], cmap='YloRd'))

