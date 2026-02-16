import streamlit as st

def render_song_card(song):
    col1, col2, col3 = st.columns([4, 3, 1])

    col1.markdown(f"🎵 **{song['title']}**")
    col2.markdown(f"🎤 {song['artist']}")

    if col3.button("▶️ Play", key=f"play_{song['title']}"):
        st.session_state.current_song = {
            "title": song["title"],
            "artist": song["artist"],
            "mood": song.get("mood", "Auto"),
            "audio": song.get("audio", "assets/audio/sample_audio.mp3")
        }
        st.success(f"Now playing: {song['title']}")
