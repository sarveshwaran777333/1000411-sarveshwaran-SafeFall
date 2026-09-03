import os
import io
import math
import wave
import tempfile

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# SAFEFALL AI
# AI-BASED VIDEO FALL DETECTION SYSTEM
# ============================================================

st.set_page_config(
    page_title="SafeFall AI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "safe_fall_model.pkl"
)

POSE_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "pose_landmarker_full.task"
)


# ============================================================
# LOAD TRAINED RANDOM FOREST
# ============================================================

@st.cache_resource
def load_model_package():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"SafeFall model not found:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# CREATE MEDIAPIPE POSE DETECTOR
# ============================================================

def create_pose_detector():

    if not os.path.exists(POSE_MODEL_PATH):
        raise FileNotFoundError(
            f"MediaPipe pose model not found:\n{POSE_MODEL_PATH}"
        )

    base_options = python.BaseOptions(
        model_asset_path=POSE_MODEL_PATH
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.25,
        min_pose_presence_confidence=0.25,
        min_tracking_confidence=0.25
    )

    return vision.PoseLandmarker.create_from_options(
        options
    )


# ============================================================
# BASE POSE FEATURES
#
# IMPORTANT:
# This matches train_model.py
# ============================================================

def calculate_features(
    landmarks,
    previous_landmarks=None
):

    if landmarks is None or len(landmarks) < 25:
        return None

    # Important body joints
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    left_hip = landmarks[23]
    right_hip = landmarks[24]

    important_points = [
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip
    ]

    # --------------------------------------------------------
    # Visibility check
    # --------------------------------------------------------

    if min(
        getattr(point, "visibility", 0.0)
        for point in important_points
    ) < 0.25:
        return None

    # --------------------------------------------------------
    # Visible pose landmarks
    # --------------------------------------------------------

    visible_points = [
        point
        for point in landmarks
        if getattr(point, "visibility", 0.0) > 0.25
    ]

    if len(visible_points) < 6:
        return None

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    xs = [
        point.x
        for point in visible_points
    ]

    ys = [
        point.y
        for point in visible_points
    ]

    min_x = min(xs)
    max_x = max(xs)

    min_y = min(ys)
    max_y = max(ys)

    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    if bbox_w <= 1e-6 or bbox_h <= 1e-6:
        return None

    # --------------------------------------------------------
    # Aspect ratio
    # --------------------------------------------------------

    aspect_ratio = bbox_w / bbox_h

    # --------------------------------------------------------
    # Shoulder midpoint
    # --------------------------------------------------------

    shoulder_x = (
        left_shoulder.x
        +
        right_shoulder.x
    ) / 2.0

    shoulder_y = (
        left_shoulder.y
        +
        right_shoulder.y
    ) / 2.0

    # --------------------------------------------------------
    # Hip midpoint
    # --------------------------------------------------------

    hip_x = (
        left_hip.x
        +
        right_hip.x
    ) / 2.0

    hip_y_raw = (
        left_hip.y
        +
        right_hip.y
    ) / 2.0

    # --------------------------------------------------------
    # Torso vector
    # --------------------------------------------------------

    torso_dx = (
        hip_x
        -
        shoulder_x
    )

    torso_dy = (
        hip_y_raw
        -
        shoulder_y
    )

    # --------------------------------------------------------
    # Torso angle
    # --------------------------------------------------------

    torso_angle = math.degrees(
        math.atan2(
            abs(torso_dy),
            abs(torso_dx) + 1e-6
        )
    )

    torso_angle = np.clip(
        torso_angle,
        0,
        90
    )

    # --------------------------------------------------------
    # Normalized hip Y
    # --------------------------------------------------------

    hip_y = (
        hip_y_raw
        -
        min_y
    ) / bbox_h

    # --------------------------------------------------------
    # Torso length ratio
    # --------------------------------------------------------

    torso_length = math.sqrt(
        torso_dx ** 2
        +
        torso_dy ** 2
    )

    torso_ratio = (
        torso_length
        /
        bbox_h
    )

    # --------------------------------------------------------
    # Motion features
    # --------------------------------------------------------

    v_hip_y = 0.0
    v_torso_angle = 0.0

    if (
        previous_landmarks is not None
        and
        len(previous_landmarks) >= 25
    ):

        prev_hip_y = (
            previous_landmarks[23].y
            +
            previous_landmarks[24].y
        ) / 2.0

        prev_shoulder_x = (
            previous_landmarks[11].x
            +
            previous_landmarks[12].x
        ) / 2.0

        prev_shoulder_y = (
            previous_landmarks[11].y
            +
            previous_landmarks[12].y
        ) / 2.0

        prev_hip_x = (
            previous_landmarks[23].x
            +
            previous_landmarks[24].x
        ) / 2.0

        prev_torso_dx = (
            prev_hip_x
            -
            prev_shoulder_x
        )

        prev_torso_dy = (
            prev_hip_y
            -
            prev_shoulder_y
        )

        prev_angle = math.degrees(
            math.atan2(
                abs(prev_torso_dy),
                abs(prev_torso_dx) + 1e-6
            )
        )

        prev_angle = np.clip(
            prev_angle,
            0,
            90
        )

        # Downward hip movement
        v_hip_y = (
            hip_y_raw
            -
            prev_hip_y
        )

        # Torso angular movement
        v_torso_angle = (
            torso_angle
            -
            prev_angle
        )

    return {

        "aspect_ratio":
            float(aspect_ratio),

        "torso_angle":
            float(torso_angle),

        "hip_y":
            float(hip_y),

        "torso_ratio":
            float(torso_ratio),

        "bbox_w":
            float(bbox_w),

        "bbox_h":
            float(bbox_h),

        "v_hip_y":
            float(v_hip_y),

        "v_torso_angle":
            float(v_torso_angle)
    }


# ============================================================
# TEMPORAL FEATURES
#
# Same rolling system as train_model.py
# ============================================================

def add_temporal_features(
    df_video,
    window_size=10
):

    df_video = df_video.copy()

    df_video = (
        df_video
        .sort_values("frame")
        .reset_index(drop=True)
    )

    # Maximum downward hip velocity
    df_video["rolling_max_v_hip"] = (
        df_video["v_hip_y"]
        .rolling(
            window=window_size,
            min_periods=1
        )
        .max()
    )

    # Mean torso angle
    df_video["rolling_mean_angle"] = (
        df_video["torso_angle"]
        .rolling(
            window=window_size,
            min_periods=1
        )
        .mean()
    )

    # Maximum body aspect ratio
    df_video["rolling_max_aspect"] = (
        df_video["aspect_ratio"]
        .rolling(
            window=window_size,
            min_periods=1
        )
        .max()
    )

    # Minimum normalized hip position
    df_video["rolling_min_hip_y"] = (
        df_video["hip_y"]
        .rolling(
            window=window_size,
            min_periods=1
        )
        .min()
    )

    # Torso angle movement range
    rolling_max_angle = (
        df_video["torso_angle"]
        .rolling(
            window=window_size,
            min_periods=1
        )
        .max()
    )

    rolling_min_angle = (
        df_video["torso_angle"]
        .rolling(
            window=window_size,
            min_periods=1
        )
        .min()
    )

    df_video["angle_change_range"] = (
        rolling_max_angle
        -
        rolling_min_angle
    )

    return df_video


# ============================================================
# CREATE ALARM SOUND
# ============================================================

@st.cache_data
def create_alarm_sound():

    sample_rate = 44100

    # About 2.4 seconds total
    duration = 2.4

    samples = int(
        sample_rate * duration
    )

    t = (
        np.arange(samples)
        /
        sample_rate
    )

    # Alternating warning frequencies
    frequency = np.where(
        (t % 0.60) < 0.30,
        880.0,
        660.0
    )

    waveform = (
        0.42
        *
        np.sin(
            2
            *
            np.pi
            *
            frequency
            *
            t
        )
    )

    # Create pulsing alarm
    pulse = (
        (t % 0.60) < 0.48
    ).astype(float)

    waveform *= pulse

    waveform = (
        waveform
        *
        32767
    ).astype(np.int16)

    buffer = io.BytesIO()

    with wave.open(
        buffer,
        "wb"
    ) as wav_file:

        wav_file.setnchannels(1)

        wav_file.setsampwidth(2)

        wav_file.setframerate(
            sample_rate
        )

        wav_file.writeframes(
            waveform.tobytes()
        )

    return buffer.getvalue()


# ============================================================
# PLAY FALL ALARM
# ============================================================

def play_fall_alarm():

    alarm = create_alarm_sound()

    try:

        st.audio(
            alarm,
            format="audio/wav",
            autoplay=True
        )

    except TypeError:

        # Compatibility fallback
        st.audio(
            alarm,
            format="audio/wav"
        )


# ============================================================
# READ ONE VIDEO FRAME
# ============================================================

def get_video_frame(
    video_path,
    frame_number
):

    capture = cv2.VideoCapture(
        video_path
    )

    if not capture.isOpened():
        return None

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        max(
            0,
            int(frame_number) - 1
        )
    )

    success, frame = capture.read()

    capture.release()

    if not success:
        return None

    return frame


# ============================================================
# DRAW RESULT ON HIGH-RISK FRAME
# ============================================================

def create_result_frame(
    frame,
    probability,
    detected
):

    result_frame = frame.copy()

    height, width = result_frame.shape[:2]

    if detected:

        text = (
            f"FALL DETECTED - "
            f"{probability * 100:.1f}%"
        )

        color = (
            0,
            0,
            255
        )

    else:

        text = (
            f"NO FALL - "
            f"{probability * 100:.1f}% risk"
        )

        color = (
            0,
            180,
            0
        )

    # Dark text background
    cv2.rectangle(
        result_frame,
        (15, 15),
        (
            min(
                width - 15,
                650
            ),
            85
        ),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        result_frame,
        text,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        3,
        cv2.LINE_AA
    )

    return cv2.cvtColor(
        result_frame,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# VIDEO FEATURE EXTRACTION
# ============================================================

def extract_video_features(
    video_path,
    progress_bar=None,
    status_box=None
):

    detector = create_pose_detector()

    capture = cv2.VideoCapture(
        video_path
    )

    if not capture.isOpened():

        detector.close()

        raise RuntimeError(
            "OpenCV could not open the uploaded video."
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if (
        fps is None
        or
        fps <= 0
        or
        not np.isfinite(fps)
    ):
        fps = 25.0

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    rows = []

    frame_number = 0

    previous_landmarks = None

    previous_timestamp = -1

    while True:

        success, frame = capture.read()

        if not success:
            break

        frame_number += 1

        # ----------------------------------------------------
        # Convert OpenCV BGR → RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # MediaPipe timestamp
        # ----------------------------------------------------

        timestamp_ms = int(
            (
                (frame_number - 1)
                /
                max(fps, 1e-6)
            )
            *
            1000
        )

        # MediaPipe VIDEO mode requires increasing timestamps
        if timestamp_ms <= previous_timestamp:

            timestamp_ms = (
                previous_timestamp
                +
                1
            )

        previous_timestamp = timestamp_ms

        # ----------------------------------------------------
        # MediaPipe image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        try:

            result = detector.detect_for_video(
                mp_image,
                timestamp_ms
            )

        except Exception:

            previous_landmarks = None

            continue

        # ----------------------------------------------------
        # No human pose found
        # ----------------------------------------------------

        if not result.pose_landmarks:

            previous_landmarks = None

        else:

            landmarks = (
                result.pose_landmarks[0]
            )

            features = calculate_features(
                landmarks,
                previous_landmarks
            )

            previous_landmarks = landmarks

            if features is not None:

                row = features.copy()

                row["frame"] = (
                    frame_number
                )

                rows.append(
                    row
                )

        # ----------------------------------------------------
        # Streamlit progress
        # ----------------------------------------------------

        if (
            progress_bar is not None
            and
            total_frames > 0
            and
            frame_number % 5 == 0
        ):

            progress = min(
                frame_number
                /
                total_frames,
                1.0
            )

            progress_bar.progress(
                progress
            )

        if (
            status_box is not None
            and
            frame_number % 20 == 0
        ):

            status_box.write(
                f"Processing frame "
                f"{frame_number:,}"
                +
                (
                    f" / {total_frames:,}"
                    if total_frames > 0
                    else ""
                )
            )

    capture.release()

    detector.close()

    if progress_bar is not None:

        progress_bar.progress(
            1.0
        )

    if not rows:

        raise RuntimeError(
            "No usable human pose could be detected "
            "in this video."
        )

    df = pd.DataFrame(
        rows
    )

    return (
        df,
        fps,
        total_frames
    )


# ============================================================
# RUN RANDOM FOREST
# ============================================================

def run_predictions(
    df,
    model,
    feature_columns,
    window_size
):

    # Same 10-frame temporal calculations as training
    df = add_temporal_features(
        df,
        window_size=window_size
    )

    # Remove invalid numerical values
    df = df.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    df = df.dropna(
        subset=feature_columns
    )

    if df.empty:

        raise RuntimeError(
            "Pose data was detected, but no valid "
            "feature rows remained for prediction."
        )

    X = df[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # Class predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X
    )

    df["prediction"] = (
        predictions.astype(int)
    )

    # --------------------------------------------------------
    # FALL probability
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(
                X
            )
        )

        model_classes = list(
            model.classes_
        )

        if 1 in model_classes:

            fall_class_index = (
                model_classes.index(1)
            )

            df["fall_probability"] = (
                probabilities[
                    :,
                    fall_class_index
                ]
            )

        else:

            df["fall_probability"] = (
                df["prediction"]
                .astype(float)
            )

    else:

        df["fall_probability"] = (
            df["prediction"]
            .astype(float)
        )

    return df


# ============================================================
# CONFIRM FALL
#
# A single noisy frame should not trigger the siren.
# Require multiple high-risk frames.
# ============================================================

def confirm_fall(
    prediction_df,
    threshold=0.50,
    minimum_frames=2
):

    risky = (
        prediction_df[
            "fall_probability"
        ]
        >=
        threshold
    )

    risk_count = int(
        risky.sum()
    )

    detected = (
        risk_count
        >=
        minimum_frames
    )

    return (
        detected,
        risk_count
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model_package = (
        load_model_package()
    )

    model = (
        model_package["model"]
    )

    feature_columns = (
        model_package[
            "feature_columns"
        ]
    )

    labels = (
        model_package[
            "labels"
        ]
    )

    accuracy = float(
        model_package[
            "accuracy"
        ]
    )

    window_size = int(
        model_package.get(
            "window_size",
            10
        )
    )

except Exception as error:

    st.error(
        "❌ SafeFall AI could not load "
        "the trained model."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# MAIN INTERFACE
# ============================================================

st.title(
    "🛡️ SafeFall AI"
)

st.subheader(
    "AI-Based Fall Detection System"
)

st.write(
    "SafeFall AI analyses human body posture and movement "
    "from video using MediaPipe Pose and a trained "
    "Random Forest machine-learning model."
)

st.success(
    "✅ SafeFall AI trained model loaded successfully!"
)


# ============================================================
# MODEL STATUS
# ============================================================

st.divider()

st.write(
    "### 🧠 AI Model Status"
)

col1, col2, col3, col4 = st.columns(
    4
)

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
        "AI Features",
        len(feature_columns)
    )

with col4:

    st.metric(
        "Temporal Window",
        f"{window_size} frames"
    )


# ============================================================
# FEATURE INFORMATION
# ============================================================

with st.expander(
    "🔬 View the 13 AI features"
):

    for number, feature in enumerate(
        feature_columns,
        start=1
    ):

        st.write(
            f"{number}. `{feature}`"
        )


# ============================================================
# VIDEO DETECTION
# ============================================================

st.divider()

st.write(
    "## 📹 Fall Detection"
)

st.write(
    "Upload a video containing a person. "
    "SafeFall AI will analyse the person's pose "
    "frame-by-frame and determine whether a fall occurred."
)

uploaded_video = st.file_uploader(
    "Upload a video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
        "mpeg",
        "mpg"
    ]
)


# ============================================================
# DETECTION SETTINGS
# ============================================================

with st.expander(
    "⚙️ Detection settings"
):

    fall_threshold = st.slider(
        "Fall probability threshold",
        min_value=0.40,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "A frame above this probability "
            "is treated as a possible fall."
        )
    )

    minimum_fall_frames = st.slider(
        "Minimum high-risk frames",
        min_value=1,
        max_value=6,
        value=2,
        step=1,
        help=(
            "Requiring multiple high-risk frames "
            "helps reduce accidental false alarms."
        )
    )


# ============================================================
# VIDEO UPLOADED
# ============================================================

if uploaded_video is not None:

    video_bytes = (
        uploaded_video.getvalue()
    )

    st.write(
        "### 🎞️ Uploaded Video"
    )

    st.video(
        video_bytes
    )

    st.write(
        f"**File:** {uploaded_video.name}"
    )

    st.write(
        f"**Size:** "
        f"{len(video_bytes) / (1024 * 1024):.2f} MB"
    )

    analyse_button = st.button(
        "🔍 Analyse Video for Falls",
        type="primary",
        use_container_width=True
    )

    if analyse_button:

        extension = os.path.splitext(
            uploaded_video.name
        )[1]

        if not extension:

            extension = ".mp4"

        temporary_path = None

        try:

            # ------------------------------------------------
            # Save uploaded video temporarily
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension
            ) as temporary_file:

                temporary_file.write(
                    video_bytes
                )

                temporary_path = (
                    temporary_file.name
                )

            st.divider()

            st.write(
                "### 🧍 Human Pose Analysis"
            )

            progress_bar = st.progress(
                0.0
            )

            status_box = st.empty()

            with st.spinner(
                "SafeFall AI is analysing the video..."
            ):

                (
                    feature_df,
                    fps,
                    total_frames
                ) = extract_video_features(
                    temporary_path,
                    progress_bar,
                    status_box
                )

                prediction_df = run_predictions(
                    feature_df,
                    model,
                    feature_columns,
                    window_size
                )

            status_box.success(
                "✅ Video analysis complete!"
            )

            # ------------------------------------------------
            # Fall confirmation
            # ------------------------------------------------

            (
                fall_detected,
                high_risk_frames
            ) = confirm_fall(
                prediction_df,
                threshold=fall_threshold,
                minimum_frames=
                    minimum_fall_frames
            )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            max_probability = float(
                prediction_df[
                    "fall_probability"
                ].max()
            )

            average_probability = float(
                prediction_df[
                    "fall_probability"
                ].mean()
            )

            highest_risk_index = (
                prediction_df[
                    "fall_probability"
                ].idxmax()
            )

            highest_risk_row = (
                prediction_df.loc[
                    highest_risk_index
                ]
            )

            highest_risk_frame = int(
                highest_risk_row[
                    "frame"
                ]
            )

            highest_risk_time = (
                (
                    highest_risk_frame
                    -
                    1
                )
                /
                fps
            )

            pose_frames = len(
                prediction_df
            )

            # =================================================
            # FALL DETECTED
            # =================================================

            st.divider()

            st.write(
                "## 🎯 SafeFall Result"
            )

            if fall_detected:

                st.error(
                    "🚨 FALL DETECTED!"
                )

                st.error(
                    "SafeFall AI detected movement "
                    "consistent with a fall in the video."
                )

                # 🔊 PLAY ALARM
                play_fall_alarm()

            else:

                st.success(
                    "✅ NO FALL DETECTED"
                )

                st.write(
                    "SafeFall AI did not detect enough "
                    "high-risk fall frames to activate "
                    "the emergency alarm."
                )

            # ------------------------------------------------
            # Result metrics
            # ------------------------------------------------

            result1, result2, result3, result4 = (
                st.columns(4)
            )

            with result1:

                st.metric(
                    "Maximum Fall Risk",
                    f"{max_probability * 100:.1f}%"
                )

            with result2:

                st.metric(
                    "High-Risk Frames",
                    high_risk_frames
                )

            with result3:

                st.metric(
                    "Pose Frames Analysed",
                    f"{pose_frames:,}"
                )

            with result4:

                st.metric(
                    "Highest-Risk Time",
                    f"{highest_risk_time:.2f} s"
                )

            # ------------------------------------------------
            # Highest-risk video frame
            # ------------------------------------------------

            st.write(
                "### 🔎 Highest-Risk Frame"
            )

            risk_frame = get_video_frame(
                temporary_path,
                highest_risk_frame
            )

            if risk_frame is not None:

                displayed_frame = (
                    create_result_frame(
                        risk_frame,
                        max_probability,
                        fall_detected
                    )
                )

                st.image(
                    displayed_frame,
                    caption=(
                        f"Frame "
                        f"{highest_risk_frame:,} "
                        f"at approximately "
                        f"{highest_risk_time:.2f} seconds"
                    ),
                    use_container_width=True
                )

            # ------------------------------------------------
            # Fall probability graph
            # ------------------------------------------------

            st.write(
                "### 📈 Fall Risk Across the Video"
            )

            chart_df = (
                prediction_df[
                    [
                        "frame",
                        "fall_probability"
                    ]
                ]
                .copy()
            )

            chart_df["Fall probability (%)"] = (
                chart_df[
                    "fall_probability"
                ]
                *
                100
            )

            chart_df = (
                chart_df
                .set_index("frame")
            )

            st.line_chart(
                chart_df[
                    [
                        "Fall probability (%)"
                    ]
                ],
                y_label="Fall probability (%)",
                x_label="Video frame"
            )

            # ------------------------------------------------
            # Detailed analysis
            # ------------------------------------------------

            with st.expander(
                "📊 View detailed frame predictions"
            ):

                display_df = (
                    prediction_df[
                        [
                            "frame",
                            "fall_probability",
                            "prediction"
                        ]
                    ]
                    .copy()
                )

                display_df[
                    "fall_probability"
                ] = (
                    display_df[
                        "fall_probability"
                    ]
                    *
                    100
                ).round(2)

                display_df[
                    "prediction"
                ] = (
                    display_df[
                        "prediction"
                    ]
                    .map(
                        {
                            0: "NOT_FALL",
                            1: "FALL"
                        }
                    )
                )

                display_df = (
                    display_df.rename(
                        columns={
                            "frame":
                                "Frame",

                            "fall_probability":
                                "Fall Probability (%)",

                            "prediction":
                                "Prediction"
                        }
                    )
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

            # ------------------------------------------------
            # Download analysis
            # ------------------------------------------------

            csv_data = (
                prediction_df
                .to_csv(
                    index=False
                )
                .encode("utf-8")
            )

            st.download_button(
                label="⬇️ Download Detection Results",
                data=csv_data,
                file_name="safefall_detection_results.csv",
                mime="text/csv",
                use_container_width=True
            )

            # ------------------------------------------------
            # Summary
            # ------------------------------------------------

            st.write(
                "### 🧠 AI Analysis Summary"
            )

            summary_col1, summary_col2 = (
                st.columns(2)
            )

            with summary_col1:

                st.write(
                    f"**Video frames:** "
                    f"{total_frames:,}"
                )

                st.write(
                    f"**Video FPS:** "
                    f"{fps:.2f}"
                )

                st.write(
                    f"**Usable pose frames:** "
                    f"{pose_frames:,}"
                )

            with summary_col2:

                st.write(
                    f"**Average fall risk:** "
                    f"{average_probability * 100:.2f}%"
                )

                st.write(
                    f"**Maximum fall risk:** "
                    f"{max_probability * 100:.2f}%"
                )

                st.write(
                    "**Final classification:** "
                    +
                    (
                        "🚨 FALL"
                        if fall_detected
                        else
                        "✅ NOT_FALL"
                    )
                )

        except Exception as error:

            st.error(
                "❌ SafeFall AI could not analyse "
                "this video."
            )

            st.exception(
                error
            )

        finally:

            if (
                temporary_path is not None
                and
                os.path.exists(
                    temporary_path
                )
            ):

                try:

                    os.remove(
                        temporary_path
                    )

                except Exception:

                    pass


# ============================================================
# SYSTEM STATUS
# ============================================================

st.divider()

st.write(
    "### 🖥️ System Status"
)

status1, status2, status3, status4 = (
    st.columns(4)
)

with status1:

    st.metric(
        "Application",
        "Ready"
    )

with status2:

    st.metric(
        "AI Model",
        "Connected"
    )

with status3:

    st.metric(
        "Pose Detection",
        "MediaPipe"
    )

with status4:

    st.metric(
        "Interface",
        "Streamlit Cloud"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SafeFall AI • AI-Based Fall Detection System • "
    "MediaPipe Pose + Random Forest"
)
