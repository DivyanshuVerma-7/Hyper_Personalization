import streamlit as st

def render_feedback():
    st.subheader("🧠 Feedback")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Like"):
            st.session_state.last_reward = 1
            st.success("Positive reward sent to RL agent")

    with col2:
        if st.button("⏭ Skip"):
            st.session_state.last_reward = -1
            st.warning("Negative reward sent to RL agent")
