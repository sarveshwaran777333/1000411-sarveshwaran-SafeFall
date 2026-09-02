import streamlit as st

st.set_page_config(
    page_title="SafeFall AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SafeFall AI")

st.subheader("AI-Based Fall Detection System")

st.write(
    "SafeFall AI is designed to detect potential falls "
    "using video-based human pose analysis and machine learning."
)

st.success("SafeFall AI application loaded successfully!")

st.info(
    "Dataset processing and the trained ML model will be connected "
    "in the next steps."
)

st.divider()

st.write("### System Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Application", "Ready")

with col2:
    st.metric("Dataset", "Kaggle")

with col3:
    st.metric("Interface", "Streamlit")
