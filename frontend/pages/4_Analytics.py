import streamlit as st
import random

st.set_page_config(layout="wide")

st.title("📊 Analytics")
st.caption("User interaction & RL signals")

st.subheader("Session Stats")

stats = {
    "Songs Played": random.randint(5, 20),
    "Songs Skipped": random.randint(1, 10),
    "Average Listening Time (sec)": random.randint(30, 180),
    "Average Heartbeat (BPM)": random.randint(70, 120),
}

col1, col2 = st.columns(2)

with col1:
    for k, v in list(stats.items())[:2]:
        st.metric(k, v)

with col2:
    for k, v in list(stats.items())[2:]:
        st.metric(k, v)

st.divider()

st.subheader("🔁 RL Reward Signals (Mock)")
st.line_chart([
    random.randint(-5, 10) for _ in range(20)
])
