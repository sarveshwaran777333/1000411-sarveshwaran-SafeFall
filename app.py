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

st.set_page_config(
    page_title="SafeFall AI",
    page_icon="🛡️",
    layout="wide"
)

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "safe_fall_model.pkl"
)

POSE_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "pose_landmarker_full.task"
)

EXPECTED_BASE_FEATURES = [
    "aspect_ratio",
    "torso_angle",
    "hip_y",
    "torso_ratio",
    "bbox_w",
    "bbox_h",
    "v_hip_y",
    "v_torso_angle"
]

EXPECTED_TEMPORAL_FEATURES = [
    "rolling_max_v_hip",
    "rolling_mean_angle",
    "rolling_max_aspect",
    "rolling_min_hip_y",
    "angle_change_range"
]

EXPECTED_FEATURE_COUNT = (
    len(EXPECTED_BASE_FEATURES)
    +
    len(EXPECTED_TEMPORAL_FEATURES)
)

@st.cache_resource
def load_model_package():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"SafeFall model not found:\n{MODEL_PATH}"
        )

    package = joblib.load(
        MODEL_PATH
    )

    required_keys = [
        "model",
        "feature_columns",
        "labels",
        "accuracy"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in package
    ]

    if missing_keys:
        raise KeyError(
            "The trained model package is missing "
            f"required keys: {missing_keys}"
        )

    return package

def create_pose_detector():

    if not os.path.exists(POSE_MODEL_PATH):
        raise FileNotFoundError(
            f"MediaPipe pose model not found:\n"
            f"{POSE_MODEL_PATH}"
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

def get_landmark_visibility(point):

    if point is None:
        return 0.0

    try:
        visibility = getattr(
            point,
            "visibility",
            None
        )

        if visibility is None:
            return 1.0

        visibility = float(
            visibility
        )

        if not np.isfinite(
            visibility
        ):
            return 0.0

        return float(
            np.clip(
                visibility,
                0.0,
                1.0
            )
        )

    except (
        TypeError,
        ValueError,
        AttributeError
    ):
        return 0.0

def landmark_coordinates_are_valid(
    point
):

    if point is None:
        return False

    try:

        x = float(
            point.x
        )

        y = float(
            point.y
        )

        return (
            np.isfinite(x)
            and
            np.isfinite(y)
        )

    except (
        TypeError,
        ValueError,
        AttributeError
    ):

        return False

def calculate_features(
    landmarks,
    previous_landmarks=None
):

    if (
        landmarks is None
        or
        len(landmarks) < 25
    ):
        return None

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

    if not all(
        landmark_coordinates_are_valid(point)
        for point in important_points
    ):
        return None

    important_visibility = [
        get_landmark_visibility(point)
        for point in important_points
    ]

    if min(
        important_visibility
    ) < 0.25:
        return None

    visible_points = []

    for point in landmarks:

        if not landmark_coordinates_are_valid(
            point
        ):
            continue

        visibility = (
            get_landmark_visibility(
                point
            )
        )

        if visibility > 0.25:
            visible_points.append(
                point
            )

    if len(
        visible_points
    ) < 6:
        return None

    xs = [
        float(point.x)
        for point in visible_points
    ]

    ys = [
        float(point.y)
        for point in visible_points
    ]

    min_x = min(xs)
    max_x = max(xs)

    min_y = min(ys)
    max_y = max(ys)

    bbox_w = (
        max_x
        -
        min_x
    )

    bbox_h = (
        max_y
        -
        min_y
    )

    if (
        bbox_w <= 1e-6
        or
        bbox_h <= 1e-6
    ):
        return None

    aspect_ratio = (
        bbox_w
        /
        bbox_h
    )

    shoulder_x = (
        float(left_shoulder.x)
        +
        float(right_shoulder.x)
    ) / 2.0

    shoulder_y = (
        float(left_shoulder.y)
        +
        float(right_shoulder.y)
    ) / 2.0

    hip_x = (
        float(left_hip.x)
        +
        float(right_hip.x)
    ) / 2.0

    hip_y_raw = (
        float(left_hip.y)
        +
        float(right_hip.y)
    ) / 2.0

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

    torso_angle = math.degrees(
        math.atan2(
            abs(torso_dy),
            abs(torso_dx) + 1e-6
        )
    )

    torso_angle = float(
        np.clip(
            torso_angle,
            0,
            90
        )
    )

    hip_y = (
        hip_y_raw
        -
        min_y
    ) / bbox_h

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

    v_hip_y = 0.0
    v_torso_angle = 0.0

    if (
        previous_landmarks is not None
        and
        len(previous_landmarks) >= 25
    ):

        previous_important_points = [
            previous_landmarks[11],
            previous_landmarks[12],
            previous_landmarks[23],
            previous_landmarks[24]
        ]

        if all(
            landmark_coordinates_are_valid(point)
            for point in previous_important_points
        ):

            prev_hip_y = (
                float(previous_landmarks[23].y)
                +
                float(previous_landmarks[24].y)
            ) / 2.0

            prev_shoulder_x = (
                float(previous_landmarks[11].x)
                +
                float(previous_landmarks[12].x)
            ) / 2.0

            prev_shoulder_y = (
                float(previous_landmarks[11].y)
                +
                float(previous_landmarks[12].y)
            ) / 2.0

            prev_hip_x = (
                float(previous_landmarks[23].x)
                +
                float(previous_landmarks[24].x)
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

            prev_angle = float(
                np.clip(
                    prev_angle,
                    0,
                    90
                )
            )

            v_hip_y = (
                hip_y_raw
                -
                prev_hip_y
            )

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

def add_temporal_features(
    df_video,
    window_size=10
):

    df_video = (
        df_video
        .copy()
    )

    df_video = (
        df_video
        .sort_values("frame")
        .reset_index(drop=True)
    )

    df_video[
        "rolling_max_v_hip"
    ] = (
        df_video[
            "v_hip_y"
        ]
        .rolling(
            window=window_size,
            min_periods=1
        )
        .max()
    )

    df_video[
        "rolling_mean_angle"
    ] = (
        df_video[
            "torso_angle"
        ]
        .rolling(
            window=window_size,
            min_periods=1
        )
        .mean()
    )

    df_video[
        "rolling_max_aspect"
    ] = (
        df_video[
            "aspect_ratio"
        ]
        .rolling(
            window=window_size,
            min_periods=1
        )
        .max()
    )

    df_video[
        "rolling_min_hip_y"
    ] = (
        df_video[
            "hip_y"
        ]
        .rolling(
            window=window_size,
            min_periods=1
        )
        .min()
    )

    rolling_max_angle = (
        df_video[
            "torso_angle"
        ]
        .rolling(
            window=window_size,
            min_periods=1
        )
        .max()
    )

    rolling_min_angle = (
        df_video[
            "torso_angle"
        ]
        .rolling(
            window=window_size,
            min_periods=1
        )
        .min()
    )

    df_video[
        "angle_change_range"
    ] = (
        rolling_max_angle
        -
        rolling_min_angle
    )

    return df_video

@st.cache_data
def create_alarm_sound():

    sample_rate = 44100

    duration = 2.4

    samples = int(
        sample_rate
        *
        duration
    )

    t = (
        np.arange(samples)
        /
        sample_rate
    )

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

    pulse = (
        (t % 0.60) < 0.48
    ).astype(float)

    waveform *= pulse

    waveform = (
        waveform
        *
        32767
    ).astype(
        np.int16
    )

    buffer = io.BytesIO()

    with wave.open(
        buffer,
        "wb"
    ) as wav_file:

        wav_file.setnchannels(
            1
        )

        wav_file.setsampwidth(
            2
        )

        wav_file.setframerate(
            sample_rate
        )

        wav_file.writeframes(
            waveform.tobytes()
        )

    return buffer.getvalue()

def play_fall_alarm():

    alarm = create_alarm_sound()

    try:

        st.audio(
            alarm,
            format="audio/wav",
            autoplay=True
        )

    except TypeError:

        st.audio(
            alarm,
            format="audio/wav"
        )

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

    success, frame = (
        capture.read()
    )

    capture.release()

    if not success:
        return None

    return frame

def create_result_frame(
    frame,
    probability,
    detected
):

    result_frame = (
        frame.copy()
    )

    height, width = (
        result_frame.shape[:2]
    )

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

def extract_video_features(
    video_path,
    progress_bar=None,
    status_box=None
):

    detector = None
    capture = None

    try:

        detector = (
            create_pose_detector()
        )

        capture = (
            cv2.VideoCapture(
                video_path
            )
        )

        if not capture.isOpened():

            raise RuntimeError(
                "OpenCV could not open "
                "the uploaded video."
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

        fps = float(
            fps
        )

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

            success, frame = (
                capture.read()
            )

            if not success:
                break

            frame_number += 1

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            timestamp_ms = int(
                (
                    (
                        frame_number - 1
                    )
                    /
                    max(
                        fps,
                        1e-6
                    )
                )
                *
                1000
            )

            if (
                timestamp_ms
                <=
                previous_timestamp
            ):

                timestamp_ms = (
                    previous_timestamp
                    +
                    1
                )

            previous_timestamp = (
                timestamp_ms
            )

            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb_frame
            )

            try:

                result = (
                    detector.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )
                )

            except Exception:

                previous_landmarks = None

                continue

            if not result.pose_landmarks:

                previous_landmarks = None

            else:

                landmarks = (
                    result.pose_landmarks[0]
                )

                features = (
                    calculate_features(
                        landmarks,
                        previous_landmarks
                    )
                )

                if features is not None:

                    previous_landmarks = (
                        landmarks
                    )

                    row = (
                        features.copy()
                    )

                    row["frame"] = (
                        frame_number
                    )

                    rows.append(
                        row
                    )

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

        if progress_bar is not None:

            progress_bar.progress(
                1.0
            )

        if not rows:

            raise RuntimeError(
                "No usable human pose could be "
                "detected in this video."
            )

        df = pd.DataFrame(
            rows
        )

        return (
            df,
            fps,
            total_frames
        )

    finally:

        if capture is not None:

            try:
                capture.release()
            except Exception:
                pass

        if detector is not None:

            try:
                detector.close()
            except Exception:
                pass

def validate_model_features(
    feature_columns
):

    if not isinstance(
        feature_columns,
        (list, tuple)
    ):

        raise TypeError(
            "The model's feature_columns must "
            "be a list or tuple."
        )

    feature_columns = list(
        feature_columns
    )

    if len(
        feature_columns
    ) != EXPECTED_FEATURE_COUNT:

        raise ValueError(
            "Model feature count mismatch.\n\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Found: {len(feature_columns)}\n\n"
            "The model and inference pipeline "
            "must use the same features."
        )

    expected_features = (
        EXPECTED_BASE_FEATURES
        +
        EXPECTED_TEMPORAL_FEATURES
    )

    missing = [
        feature
        for feature in expected_features
        if feature not in feature_columns
    ]

    unexpected = [
        feature
        for feature in feature_columns
        if feature not in expected_features
    ]

    if missing or unexpected:

        raise ValueError(
            "Model feature definitions do not match "
            "the SafeFall inference pipeline.\n\n"
            f"Missing features: {missing}\n"
            f"Unexpected features: {unexpected}"
        )

    return feature_columns

def get_fall_class(
    model,
    labels
):

    model_classes = list(
        getattr(
            model,
            "classes_",
            []
        )
    )

    if 1 in model_classes:

        return 1

    if labels is not None:

        try:

            for label in labels:

                try:

                    if int(label) == 1:

                        return label

                except (
                    TypeError,
                    ValueError
                ):

                    continue

        except TypeError:

            pass

    return 1

def run_predictions(
    df,
    model,
    feature_columns,
    window_size,
    labels=None
):

    feature_columns = (
        validate_model_features(
            feature_columns
        )
    )

    df = add_temporal_features(
        df,
        window_size=window_size
    )

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

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise KeyError(
            "Missing model features during prediction: "
            f"{missing_columns}"
        )

    X = df[
        feature_columns
    ].copy()

    predictions = (
        model.predict(X)
    )

    df["prediction"] = (
        predictions
    )

    fall_class = (
        get_fall_class(
            model,
            labels
        )
    )

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

        if fall_class in model_classes:

            fall_class_index = (
                model_classes.index(
                    fall_class
                )
            )

            df[
                "fall_probability"
            ] = (
                probabilities[
                    :,
                    fall_class_index
                ]
            )

        elif 1 in model_classes:

            fall_class_index = (
                model_classes.index(1)
            )

            df[
                "fall_probability"
            ] = (
                probabilities[
                    :,
                    fall_class_index
                ]
            )

        else:

            df[
                "fall_probability"
            ] = (
                np.asarray(
                    predictions
                )
                ==
                fall_class
            ).astype(
                float
            )

    else:

        df[
            "fall_probability"
        ] = (
            np.asarray(
                predictions
            )
            ==
            fall_class
        ).astype(
            float
        )

    df[
        "fall_probability"
    ] = pd.to_numeric(
        df[
            "fall_probability"
        ],
        errors="coerce"
    )

    df[
        "fall_probability"
    ] = (
        df[
            "fall_probability"
        ]
        .clip(
            0.0,
            1.0
        )
    )

    df = df.dropna(
        subset=[
            "fall_probability"
        ]
    )

    if df.empty:

        raise RuntimeError(
            "The model produced no valid fall "
            "probability values."
        )

    return df

def confirm_fall(
    prediction_df,
    threshold=0.50,
    minimum_frames=2,
    max_frame_gap=2
):

    if prediction_df.empty:

        return (
            False,
            0
        )

    df = (
        prediction_df
        .sort_values("frame")
        .reset_index(drop=True)
    )

    risky_frames = (
        df[
            df[
                "fall_probability"
            ]
            >=
            threshold
        ][
            "frame"
        ]
        .astype(int)
        .tolist()
    )

    if not risky_frames:

        return (
            False,
            0
        )

    if minimum_frames <= 1:

        return (
            True,
            len(risky_frames)
        )

    longest_cluster = 1
    current_cluster = 1

    for index in range(
        1,
        len(risky_frames)
    ):

        frame_gap = (
            risky_frames[index]
            -
            risky_frames[index - 1]
        )

        if frame_gap <= max_frame_gap:

            current_cluster += 1

        else:

            current_cluster = 1

        longest_cluster = max(
            longest_cluster,
            current_cluster
        )

    detected = (
        longest_cluster
        >=
        minimum_frames
    )

    return (
        detected,
        longest_cluster
    )

try:

    model_package = (
        load_model_package()
    )

    model = (
        model_package[
            "model"
        ]
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

    feature_columns = (
        validate_model_features(
            feature_columns
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
            "The required number of nearby high-risk "
            "frames helps reduce accidental false alarms."
        )
    )

    max_frame_gap = st.slider(
        "Maximum frame gap between high-risk frames",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        help=(
            "High-risk frames separated by more than "
            "this many video frames start a new risk cluster."
        )
    )

if uploaded_video is not None:

    video_bytes = uploaded_video.getvalue()

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
        )[1].lower()

        if not extension:
            extension = ".avi"

        temporary_path = None

        try:

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
                    window_size,
                    labels
                )

            status_box.success(
                "✅ Video analysis complete!"
            )

            (
                fall_detected,
                high_risk_frames
            ) = confirm_fall(
                prediction_df,
                threshold=fall_threshold,
                minimum_frames=(
                    minimum_fall_frames
                ),
                max_frame_gap=max_frame_gap
            )

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

                play_fall_alarm()

            else:

                st.success(
                    "✅ NO FALL DETECTED"
                )

                st.write(
                    "SafeFall AI did not detect enough "
                    "nearby high-risk fall frames to "
                    "activate the emergency alarm."
                )

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

            chart_df[
                "Fall probability (%)"
            ] = (
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

                def display_prediction(
                    prediction
                ):

                    try:

                        if int(prediction) == 1:
                            return "FALL"

                        if int(prediction) == 0:
                            return "NOT_FALL"

                    except (
                        TypeError,
                        ValueError
                    ):
                        pass

                    return str(
                        prediction
                    )

                display_df[
                    "prediction"
                ] = (
                    display_df[
                        "prediction"
                    ]
                    .apply(
                        display_prediction
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
                file_name=(
                    "safefall_detection_results.csv"
                ),
                mime="text/csv",
                use_container_width=True
            )

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
                    f"**Average model-estimated "
                    f"fall probability:** "
                    f"{average_probability * 100:.2f}%"
                )

                st.write(
                    f"**Maximum model-estimated "
                    f"fall probability:** "
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

st.divider()

st.caption(
    "SafeFall AI • AI-Based Fall Detection System • "
    "MediaPipe Pose + Random Forest"
)
