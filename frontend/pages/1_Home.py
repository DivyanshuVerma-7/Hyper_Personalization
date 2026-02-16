import streamlit as st

st.set_page_config(layout="wide")

st.title("🏠 Home")
st.caption("Context-aware music experience")

# Initialize session state
if "current_song" not in st.session_state:
    st.session_state.current_song = {
        "title": "Midnight Vibes",
        "artist": "Neural Beats",
        "mood": "Chill",
        "audio": "assets/audio/sample_audio.mp3"
    }

col1, col2 = st.columns([1, 3])

with col1:
    st.image(
        "assets/images/default_album.png",
        width=180
    )

with col2:
    song = st.session_state.current_song
    st.subheader(song["title"])
    st.write(f"🎤 {song['artist']}")
    st.write(f"🎧 Mood: {song['mood']}")
    st.audio(song["audio"])

st.divider()

st.subheader("📌 Current Context")
st.json({
    "location": st.session_state.get("location", "Unknown"),
    "activity": st.session_state.get("activity", "Unknown"),
    "heartbeat": st.session_state.get("heartbeat", "N/A"),
    "mood": st.session_state.get("mood", "Auto")
})
