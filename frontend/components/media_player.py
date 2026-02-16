import streamlit as st

def render_media_player(song):
    if not song:
        st.warning("No song selected")
        return

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(
            "assets/images/default_album.png",
            width=180
        )

    with col2:
        st.subheader(song["title"])
        st.write(f"🎤 Artist: {song['artist']}")
        st.write(f"🎧 Mood: {song.get('mood', 'Auto')}")
        st.audio(song["audio"])
