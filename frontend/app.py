import streamlit as st

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Context-Aware Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# ---------------- Session State Init ----------------
def init_session_state():
    defaults = {
        # Context
        "location": "Unknown",
        "activity": "Relaxing",
        "heartbeat": 80,
        "mood": "Auto",

        # Playback
        "current_song": {
            "title": "Midnight Vibes",
            "artist": "Neural Beats",
            "mood": "Chill",
            "audio": "assets/audio/sample_audio.mp3"
        },

        # RL-related
        "refresh": False,
        "last_reward": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# ---------------- Sidebar (Component) ----------------
from components.sidebar import render_sidebar
render_sidebar()

# ---------------- Main Landing Content ----------------
st.title("🎵 Context-Aware Reinforcement Learning Based Music Recommendation System")

st.markdown(
    """
    This system recommends music using:
    
    - **User context** (location, activity)
    - **Physiological signals** (heartbeat / BPM)
    - **Implicit & explicit feedback** (play, skip, like)
    - **PPO-based Reinforcement Learning policy**
    
    The frontend collects signals and sends them to the RL agent
    for inference and reward learning.
    """
)

st.info("👈 Use the sidebar to personalize your music experience")

# ---------------- System Status ----------------
st.subheader("📡 Current Context Snapshot")

col1, col2, col3, col4 = st.columns(4)

col1.metric("📍 Location", st.session_state.location)
col2.metric("🏃 Activity", st.session_state.activity)
col3.metric("❤️ Heartbeat (BPM)", st.session_state.heartbeat)
col4.metric("😊 Mood", st.session_state.mood)

# ---------------- RL Debug (Optional) ----------------
with st.expander("🔧 Debug: Session State (RL Input)"):
    st.json({
        "location": st.session_state.location,
        "activity": st.session_state.activity,
        "heartbeat": st.session_state.heartbeat,
        "mood": st.session_state.mood,
        "last_reward": st.session_state.last_reward,
        "refresh": st.session_state.refresh,
    })
