import streamlit as st

def render_sidebar():
    st.sidebar.title("🎧 Context Inputs")

    st.session_state.location = st.sidebar.selectbox(
        "📍 Location",
        ["New York", "London", "Berlin", "Mumbai"],
        index=0
    )

    st.session_state.activity = st.sidebar.selectbox(
        "🏃 Activity",
        ["Relaxing", "Walking", "Gym", "Studying"],
        index=0
    )

    st.session_state.heartbeat = st.sidebar.slider(
        "❤️ Heartbeat (BPM)",
        min_value=50,
        max_value=180,
        value=st.session_state.get("heartbeat", 80)
    )

    st.session_state.mood = st.sidebar.selectbox(
        "😊 Mood",
        ["Auto", "Happy", "Calm", "Energetic", "Sad"],
        index=0
    )

    if st.sidebar.button("🔄 Refresh Recommendations"):
        st.session_state.refresh = True

    st.sidebar.divider()
    st.sidebar.caption("Context-Aware RL Music System")
