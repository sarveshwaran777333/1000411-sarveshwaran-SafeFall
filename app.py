import os
import joblib
import streamlit as st

# --------------------------------------------------
# SafeFall AI - Application
# --------------------------------------------------

st.set_page_config(
    page_title="SafeFall AI",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "safe_fall_model.pkl"
)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at:\n{MODEL_PATH}"
        )

    package = joblib.load(MODEL_PATH)
    return package


# --------------------------------------------------
# Interface
# --------------------------------------------------

st.title("🛡️ SafeFall AI")

st.subheader("AI-Based Fall Detection System")

st.write(
    "SafeFall AI is designed to detect potential falls "
    "using video-based human pose analysis and machine learning."
)

# --------------------------------------------------
# Load AI model
# --------------------------------------------------

try:
    model_package = load_model()

    model = model_package["model"]
    feature_columns = model_package["feature_columns"]
    labels = model_package["labels"]
    accuracy = model_package["accuracy"]

    st.success("✅ SafeFall AI trained model loaded successfully!")

except Exception as e:
    st.error("❌ Failed to load the SafeFall AI model.")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# Model information
# --------------------------------------------------

st.divider()

st.write("### 🧠 AI Model Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "Random Forest"
    )

with col2:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col3:
    st.metric(
        "Features",
        len(feature_columns)
    )

# --------------------------------------------------
# Prediction labels
# --------------------------------------------------

st.divider()

st.write("### 🎯 Prediction Classes")

col1, col2 = st.columns(2)

with col1:
    st.info(
        f"**Class 0:** {labels.get(0, 'NOT_FALL')}"
    )

with col2:
    st.warning(
        f"**Class 1:** {labels.get(1, 'FALL')}"
    )

# --------------------------------------------------
# Feature information
# --------------------------------------------------

with st.expander("View AI features"):
    for feature in feature_columns:
        st.write(f"• `{feature}`")

st.divider()

st.write("### 📹 Fall Detection")

st.info(
    "The trained SafeFall AI model is now connected to the application. "
    "Video-based prediction will be added in the next step."
)

# --------------------------------------------------
# System Status
# --------------------------------------------------

st.write("### System Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Application", "Ready")

with col2:
    st.metric("AI Model", "Connected")

with col3:
    st.metric("Interface", "Streamlit")
