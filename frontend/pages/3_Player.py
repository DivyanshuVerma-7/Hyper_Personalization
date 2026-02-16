import streamlit as st

st.set_page_config(layout="wide")

st.title("▶️ Player")

if "current_song" not in st.session_state:
    st.warning("No song selected yet.")
    st.stop()

song = st.session_state.current_song

col1, col2 = st.columns([1, 3])

with col1:
    st.image(
        "assets/images/default_album.png",
        width=200
    )

with col2:
    st.subheader(song["title"])
    st.write(f"🎤 Artist: {song['artist']}")
    st.write(f"🎼 Mood: {song['mood']}")
    st.audio(song["audio"])

st.divider()

st.subheader("🧠 Feedback")
col1, col2 = st.columns(2)

with col1:
    if st.button("👍 Like"):
        st.success("Positive reward sent to RL agent")

with col2:
    if st.button("⏭ Skip"):
        st.warning("Negative reward sent to RL agent")
