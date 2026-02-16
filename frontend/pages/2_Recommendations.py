import streamlit as st

st.set_page_config(layout="wide")

st.title("🔥 Recommendations")
st.caption("Generated using contextual signals")

# Mock recommendations
recommendations = [
    {"title": "Pulse Drive", "artist": "RL Waves"},
    {"title": "Calm Signals", "artist": "Deep Context"},
    {"title": "Heartbeat Sync", "artist": "BioGroove"},
    {"title": "Night Runner", "artist": "Urban Tempo"},
]

for song in recommendations:
    with st.container():
        col1, col2, col3 = st.columns([3, 2, 1])

        col1.markdown(f"🎵 **{song['title']}**")
        col2.markdown(f"🎤 {song['artist']}")

        if col3.button("▶️ Play", key=song["title"]):
            st.session_state.current_song = {
                "title": song["title"],
                "artist": song["artist"],
                "mood": "Auto",
                "audio": "assets/audio/sample_audio.mp3"
            }
            st.success(f"Now playing: {song['title']}")
