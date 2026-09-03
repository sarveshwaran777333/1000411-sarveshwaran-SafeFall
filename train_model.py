import os
import re
import glob
import math
import subprocess
import warnings

import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# MEDIA PIPE LOG SETTINGS
# ============================================================

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = r"D:\AI\ib_ai(a)\year 2\archive"

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "safe_fall_model.pkl"
)

FEATURE_CSV = os.path.join(
    MODEL_DIR,
    "training_features.csv"
)

EXTRACTED_FRAMES_DIR = os.path.join(
    MODEL_DIR,
    "extracted_frames"
)

POSE_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "pose_landmarker_full.task"
)

# ------------------------------------------------------------
# Training folders
# ------------------------------------------------------------

TRAINING_FOLDERS = [
    "Coffee_room_01",
    "Coffee_room_02",
    "Home_01",
    "Home_02"
]

# ============================================================
# FEATURE COLUMNS
# ============================================================

BASE_FEATURE_COLUMNS = [

    "aspect_ratio",

    "torso_angle",

    "hip_y",

    "torso_ratio",

    "bbox_w",

    "bbox_h",

    "v_hip_y",

    "v_torso_angle"
]

TEMPORAL_FEATURE_COLUMNS = [

    "rolling_max_v_hip",

    "rolling_mean_angle",

    "rolling_max_aspect",

    "rolling_min_hip_y",

    "angle_change_range"
]

FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS
    +
    TEMPORAL_FEATURE_COLUMNS
)


# ============================================================
# DATASET CHECK
# ============================================================

def check_dataset():

    print()
    print("=" * 75)
    print("CHECKING DATASET")
    print("=" * 75)

    if not os.path.isdir(DATASET_ROOT):

        raise FileNotFoundError(
            f"\nDataset folder not found:\n{DATASET_ROOT}"
        )

    print(
        f"Dataset root:\n{DATASET_ROOT}"
    )

    print()

    for folder in TRAINING_FOLDERS:

        folder_path = os.path.join(
            DATASET_ROOT,
            folder
        )

        if os.path.isdir(folder_path):

            print(
                f"[OK] {folder}"
            )

        else:

            print(
                f"[WARNING] Missing folder: {folder}"
            )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        EXTRACTED_FRAMES_DIR,
        exist_ok=True
    )


# ============================================================
# CREATE MEDIAPIPE POSE DETECTOR
# ============================================================

def create_pose_detector():

    print()
    print("=" * 75)
    print("LOADING MEDIAPIPE POSE MODEL")
    print("=" * 75)

    if not os.path.isfile(
        POSE_MODEL_PATH
    ):

        raise FileNotFoundError(
            "\npose_landmarker_full.task was not found:\n"
            + POSE_MODEL_PATH
        )

    print(
        f"[OK] Pose model:\n{POSE_MODEL_PATH}"
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

    detector = (
        vision.PoseLandmarker
        .create_from_options(options)
    )

    return detector


# ============================================================
# FIND ALL TRAINING VIDEOS
# ============================================================

def find_training_videos():

    print()
    print("=" * 75)
    print("SEARCHING FOR TRAINING VIDEOS")
    print("=" * 75)

    videos = []

    for folder in TRAINING_FOLDERS:

        folder_path = os.path.join(
            DATASET_ROOT,
            folder
        )

        if not os.path.isdir(
            folder_path
        ):

            continue

        found = glob.glob(
            os.path.join(
                folder_path,
                "**",
                "*.avi"
            ),
            recursive=True
        )

        found = sorted(found)

        print(
            f"{folder}: {len(found)} video(s)"
        )

        videos.extend(found)

    # --------------------------------------------------------
    # Remove duplicate absolute paths
    # --------------------------------------------------------

    unique_videos = []

    seen = set()

    for video in videos:

        normalized = os.path.normcase(
            os.path.abspath(video)
        )

        if normalized not in seen:

            seen.add(normalized)

            unique_videos.append(
                video
            )

    print()
    print(
        f"TOTAL UNIQUE VIDEOS: "
        f"{len(unique_videos)}"
    )

    return unique_videos


# ============================================================
# CREATE UNIQUE VIDEO ID
# ============================================================

def get_video_identity(video_path):

    """
    Example:

    D:\...\Coffee_room_01\video (1).avi

    becomes:

    Coffee_room_01__video (1)
    """

    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    # Find which training folder the video belongs to
    relative_path = os.path.relpath(
        video_path,
        DATASET_ROOT
    )

    relative_parts = (
        relative_path
        .replace("\\", "/")
        .split("/")
    )

    if len(relative_parts) >= 2:

        dataset_folder = relative_parts[0]

    else:

        dataset_folder = (
            os.path.basename(
                os.path.dirname(video_path)
            )
        )

    video_id = (
        f"{dataset_folder}__{video_name}"
    )

    return (
        dataset_folder,
        video_name,
        video_id
    )


# ============================================================
# FIND CORRECT ANNOTATION
# ============================================================

def find_annotation(video_path):

    """
    Annotation matching priority:

    1. Same folder as video
    2. Subfolders of video's folder
    3. Same training dataset folder
    4. Entire dataset as final fallback

    This prevents:

        Coffee_room_01/video (1).avi

    from accidentally using:

        Home_01/video (1).txt
    """

    (
        dataset_folder,
        video_name,
        video_id
    ) = get_video_identity(
        video_path
    )

    target_name = (
        video_name + ".txt"
    )

    video_directory = os.path.dirname(
        video_path
    )

    # --------------------------------------------------------
    # 1. SAME DIRECTORY
    # --------------------------------------------------------

    same_directory = os.path.join(
        video_directory,
        target_name
    )

    if os.path.isfile(
        same_directory
    ):

        return same_directory

    # --------------------------------------------------------
    # 2. SEARCH BELOW VIDEO DIRECTORY
    # --------------------------------------------------------

    local_matches = glob.glob(
        os.path.join(
            video_directory,
            "**",
            target_name
        ),
        recursive=True
    )

    if local_matches:

        return local_matches[0]

    # --------------------------------------------------------
    # 3. SEARCH ONLY WITHIN SAME DATASET FOLDER
    # --------------------------------------------------------

    dataset_folder_path = os.path.join(
        DATASET_ROOT,
        dataset_folder
    )

    dataset_matches = glob.glob(
        os.path.join(
            dataset_folder_path,
            "**",
            target_name
        ),
        recursive=True
    )

    if dataset_matches:

        return dataset_matches[0]

    # --------------------------------------------------------
    # 4. GLOBAL FALLBACK
    # --------------------------------------------------------

    global_matches = glob.glob(
        os.path.join(
            DATASET_ROOT,
            "**",
            target_name
        ),
        recursive=True
    )

    if global_matches:

        print(
            "[WARNING] Annotation was found "
            "outside the video's dataset folder."
        )

        print(
            f"Video: {video_path}"
        )

        print(
            f"Using: {global_matches[0]}"
        )

        return global_matches[0]

    return None


# ============================================================
# READ ANNOTATION
# ============================================================

def read_annotation(annotation_path):

    """
    Handles annotations where:

    1
    100

    OR

    text
    text
    1
    100

    OR

    other annotation data
    50
    80

    Code-2 compatible parsing.
    """

    if annotation_path is None:

        return None, None

    try:

        with open(
            annotation_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            lines = [
                line.strip()
                for line in file
                if line.strip()
            ]

    except Exception as error:

        print(
            f"[ERROR] Could not read annotation: "
            f"{error}"
        )

        return None, None

    # --------------------------------------------------------
    # Find standalone integer lines
    # --------------------------------------------------------

    numbers = []

    for index, line in enumerate(lines):

        if re.fullmatch(
            r"\d+",
            line
        ):

            numbers.append(
                (
                    index,
                    int(line)
                )
            )

    # Need at least two frame numbers
    if len(numbers) < 2:

        return None, None

    # --------------------------------------------------------
    # Code-2 logic
    # --------------------------------------------------------

    if numbers[0][0] == 0:

        start_frame = numbers[0][1]

        end_frame = numbers[1][1]

    else:

        start_frame = numbers[-2][1]

        end_frame = numbers[-1][1]

    return (
        start_frame,
        end_frame
    )


# ============================================================
# GET VIDEO INFORMATION
# ============================================================

def get_video_info(video_path):

    command = [

        "ffprobe",

        "-v",
        "error",

        "-select_streams",
        "v:0",

        "-show_entries",
        "stream=width,height,r_frame_rate,nb_read_frames",

        "-of",
        "csv=p=0",

        "-count_frames",

        video_path
    ]

    try:

        result = subprocess.run(

            command,

            capture_output=True,

            text=True
        )

        output = result.stdout.strip()

        if not output:

            return None

        parts = output.split(",")

        if len(parts) < 4:

            return None

        width = int(
            parts[0]
        )

        height = int(
            parts[1]
        )

        fps_string = parts[2]

        if "/" in fps_string:

            numerator, denominator = (
                fps_string.split("/")
            )

            numerator = float(
                numerator
            )

            denominator = float(
                denominator
            )

            if denominator == 0:

                return None

            fps = (
                numerator /
                denominator
            )

        else:

            fps = float(
                fps_string
            )

        frame_count = int(
            parts[3]
        )

        return {

            "width": width,

            "height": height,

            "fps": fps,

            "frame_count": frame_count
        }

    except Exception as error:

        print(
            f"[ERROR] ffprobe failed: {error}"
        )

        return None


# ============================================================
# CALCULATE POSE FEATURES
# ============================================================

def calculate_features(
    landmarks,
    previous_landmarks=None
):

    if (
        landmarks is None
        or len(landmarks) < 25
    ):

        return None

    # --------------------------------------------------------
    # Important joints
    # --------------------------------------------------------

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

        getattr(
            point,
            "visibility",
            0.0
        )

        for point in important_points

    ) < 0.25:

        return None

    # --------------------------------------------------------
    # Visible landmarks
    # --------------------------------------------------------

    visible_points = [

        point

        for point in landmarks

        if getattr(
            point,
            "visibility",
            0.0
        ) > 0.25
    ]

    if len(
        visible_points
    ) < 6:

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

    bbox_w = (
        max_x -
        min_x
    )

    bbox_h = (
        max_y -
        min_y
    )

    if (
        bbox_w <= 1e-6
        or bbox_h <= 1e-6
    ):

        return None

    # --------------------------------------------------------
    # Aspect ratio
    # --------------------------------------------------------

    aspect_ratio = (
        bbox_w /
        bbox_h
    )

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
        hip_x -
        shoulder_x
    )

    torso_dy = (
        hip_y_raw -
        shoulder_y
    )

    # --------------------------------------------------------
    # Torso angle
    # --------------------------------------------------------

    torso_angle = math.degrees(

        math.atan2(

            abs(torso_dy),

            abs(torso_dx)
            + 1e-6
        )
    )

    torso_angle = np.clip(

        torso_angle,

        0,

        90
    )

    # --------------------------------------------------------
    # Normalized hip position
    # --------------------------------------------------------

    hip_y = (

        hip_y_raw -
        min_y

    ) / bbox_h

    # --------------------------------------------------------
    # Torso length
    # --------------------------------------------------------

    torso_length = math.sqrt(

        torso_dx ** 2
        +
        torso_dy ** 2
    )

    torso_ratio = (

        torso_length /
        bbox_h
    )

    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

    v_hip_y = 0.0

    v_torso_angle = 0.0

    if (

        previous_landmarks is not None

        and

        len(previous_landmarks) >= 25

    ):

        # Previous hip
        prev_hip_y = (

            previous_landmarks[23].y
            +
            previous_landmarks[24].y

        ) / 2.0

        # Previous shoulder
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

        # Previous hip x
        prev_hip_x = (

            previous_landmarks[23].x
            +
            previous_landmarks[24].x

        ) / 2.0

        prev_torso_dx = (

            prev_hip_x -
            prev_shoulder_x
        )

        prev_torso_dy = (

            prev_hip_y -
            prev_shoulder_y
        )

        prev_angle = math.degrees(

            math.atan2(

                abs(prev_torso_dy),

                abs(prev_torso_dx)
                + 1e-6
            )
        )

        prev_angle = np.clip(

            prev_angle,

            0,

            90
        )

        # Hip vertical velocity
        v_hip_y = (

            hip_y_raw -
            prev_hip_y
        )

        # Torso angular velocity
        v_torso_angle = (

            torso_angle -
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
# ADD TEMPORAL FEATURES
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

    # --------------------------------------------------------
    # Maximum hip downward velocity
    # --------------------------------------------------------

    df_video[
        "rolling_max_v_hip"
    ] = (

        df_video["v_hip_y"]

        .rolling(

            window=window_size,

            min_periods=1
        )

        .max()
    )

    # --------------------------------------------------------
    # Mean torso angle
    # --------------------------------------------------------

    df_video[
        "rolling_mean_angle"
    ] = (

        df_video["torso_angle"]

        .rolling(

            window=window_size,

            min_periods=1
        )

        .mean()
    )

    # --------------------------------------------------------
    # Maximum aspect ratio
    # --------------------------------------------------------

    df_video[
        "rolling_max_aspect"
    ] = (

        df_video["aspect_ratio"]

        .rolling(

            window=window_size,

            min_periods=1
        )

        .max()
    )

    # --------------------------------------------------------
    # Minimum hip position
    # --------------------------------------------------------

    df_video[
        "rolling_min_hip_y"
    ] = (

        df_video["hip_y"]

        .rolling(

            window=window_size,

            min_periods=1
        )

        .min()
    )

    # --------------------------------------------------------
    # Torso angle range
    # --------------------------------------------------------

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

    df_video[
        "angle_change_range"
    ] = (

        rolling_max_angle
        -
        rolling_min_angle
    )

    return df_video


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_path,
    annotation_path,
    detector,
    save_frames=True
):

    (
        dataset_folder,
        video_name,
        video_id
    ) = get_video_identity(
        video_path
    )

    print()
    print("-" * 75)

    print(
        f"DATASET FOLDER : {dataset_folder}"
    )

    print(
        f"VIDEO NAME     : {video_name}"
    )

    print(
        f"UNIQUE VIDEO ID: {video_id}"
    )

    print(
        f"VIDEO PATH     : {video_path}"
    )

    # --------------------------------------------------------
    # Annotation
    # --------------------------------------------------------

    if annotation_path is None:

        print(
            "\n[WARNING] NO ANNOTATION FOUND"
        )

        print(
            "This video will be treated as NOT_FALL."
        )

        fall_start = 0

        fall_end = 0

    else:

        print(
            f"\nANNOTATION:\n{annotation_path}"
        )

        (
            fall_start,
            fall_end
        ) = read_annotation(
            annotation_path
        )

        if (

            fall_start is None

            or

            fall_end is None

        ):

            print(
                "[WARNING] INVALID ANNOTATION"
            )

            fall_start = 0

            fall_end = 0

        else:

            print(
                f"FALL START: {fall_start}"
            )

            print(
                f"FALL END  : {fall_end}"
            )

    # --------------------------------------------------------
    # Video information
    # --------------------------------------------------------

    info = get_video_info(
        video_path
    )

    if info is None:

        print(
            "[ERROR] Could not read video information."
        )

        return []

    width = info["width"]

    height = info["height"]

    fps = info["fps"]

    frame_count = info["frame_count"]

    print()
    print(
        f"VIDEO INFO: "
        f"{width}x{height} | "
        f"{fps:.2f} FPS | "
        f"{frame_count} frames"
    )

    # --------------------------------------------------------
    # Correct fall range
    # --------------------------------------------------------

    if fall_start > 0:

        if fall_end < fall_start:

            print(
                "[WARNING] Fall end is before fall start."
            )

            fall_start = 0

            fall_end = 0

        else:

            fall_end = min(
                fall_end,
                frame_count
            )

    # --------------------------------------------------------
    # Unique output folders
    # --------------------------------------------------------

    if save_frames:

        fall_dir = os.path.join(

            EXTRACTED_FRAMES_DIR,

            "FALL",

            video_id
        )

        notfall_dir = os.path.join(

            EXTRACTED_FRAMES_DIR,

            "NOT_FALL",

            video_id
        )

        os.makedirs(

            fall_dir,

            exist_ok=True
        )

        os.makedirs(

            notfall_dir,

            exist_ok=True
        )

    # --------------------------------------------------------
    # FFmpeg
    # --------------------------------------------------------

    command = [

        "ffmpeg",

        "-loglevel",
        "error",

        "-an",

        "-i",
        video_path,

        "-f",
        "image2pipe",

        "-pix_fmt",
        "rgb24",

        "-vcodec",
        "rawvideo",

        "-"
    ]

    process = subprocess.Popen(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE
    )

    frame_size = (

        width *
        height *
        3
    )

    rows = []

    previous_landmarks = None

    decoded_frames = 0

    pose_frames = 0

    fall_rows = 0

    notfall_rows = 0

    # --------------------------------------------------------
    # Process every frame
    # --------------------------------------------------------

    while True:

        raw_frame = process.stdout.read(
            frame_size
        )

        if len(raw_frame) != frame_size:

            break

        decoded_frames += 1

        frame_number = decoded_frames

        frame_array = (

            np.frombuffer(

                raw_frame,

                dtype=np.uint8

            )

            .reshape(

                (
                    height,
                    width,
                    3
                )
            )
        )

        # ----------------------------------------------------
        # LABEL USING ANNOTATION
        # ----------------------------------------------------

        if (

            fall_start == 0

            and

            fall_end == 0

        ):

            label = 0

        elif (

            fall_start
            <=
            frame_number
            <=
            fall_end

        ):

            label = 1

        else:

            label = 0

        # ----------------------------------------------------
        # SAVE FRAME
        # ----------------------------------------------------

        if save_frames:

            if label == 1:

                output_path = os.path.join(

                    fall_dir,

                    f"frame_{frame_number:06d}.jpg"
                )

            else:

                output_path = os.path.join(

                    notfall_dir,

                    f"frame_{frame_number:06d}.jpg"
                )

            try:

                from PIL import Image

                Image.fromarray(
                    frame_array
                ).save(

                    output_path,

                    quality=90
                )

            except Exception as error:

                print(
                    f"[WARNING] Could not save frame "
                    f"{frame_number}: {error}"
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

        # MediaPipe requires strictly increasing timestamps
        if hasattr(
            process_video,
            "_last_timestamp"
        ):

            if (

                timestamp_ms
                <=
                process_video._last_timestamp

            ):

                timestamp_ms = (

                    process_video._last_timestamp
                    +
                    1
                )

        process_video._last_timestamp = (
            timestamp_ms
        )

        # ----------------------------------------------------
        # MediaPipe image
        # ----------------------------------------------------

        mp_image = mp.Image(

            image_format=
            mp.ImageFormat.SRGB,

            data=frame_array
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
        # No pose
        # ----------------------------------------------------

        if not result.pose_landmarks:

            previous_landmarks = None

            continue

        landmarks = (
            result.pose_landmarks[0]
        )

        # ----------------------------------------------------
        # Calculate features
        # ----------------------------------------------------

        features = calculate_features(

            landmarks,

            previous_landmarks
        )

        previous_landmarks = landmarks

        if features is None:

            continue

        pose_frames += 1

        row = features.copy()

        row["label"] = label

        # IMPORTANT:
        # Full path gives unique grouping.
        row["video"] = os.path.abspath(
            video_path
        )

        # Unique readable ID.
        row["video_name"] = video_id

        row["dataset_folder"] = (
            dataset_folder
        )

        row["frame"] = frame_number

        rows.append(row)

        if label == 1:

            fall_rows += 1

        else:

            notfall_rows += 1

    # --------------------------------------------------------
    # Close FFmpeg
    # --------------------------------------------------------

    process.stdout.close()

    process.wait()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()

    print(
        f"DECODED FRAMES : {decoded_frames}"
    )

    print(
        f"POSE FRAMES    : {pose_frames}"
    )

    print(
        f"FALL ROWS      : {fall_rows}"
    )

    print(
        f"NOT-FALL ROWS  : {notfall_rows}"
    )

    if save_frames:

        print(
            f"\nFALL FRAMES:\n{fall_dir}"
        )

        print(
            f"\nNOT-FALL FRAMES:\n{notfall_dir}"
        )

    return rows


# ============================================================
# EXTRACT FEATURES FROM ALL VIDEOS
# ============================================================

def extract_all_features(
    detector
):

    print()
    print("=" * 75)
    print("ANNOTATION-DRIVEN FEATURE EXTRACTION")
    print("=" * 75)

    videos = find_training_videos()

    if not videos:

        raise RuntimeError(
            "No AVI videos were found."
        )

    all_rows = []

    processed_videos = 0

    failed_videos = 0

    # --------------------------------------------------------
    # Process every video
    # --------------------------------------------------------

    for index, video_path in enumerate(

        videos,

        start=1
    ):

        print()

        print(
            f"[VIDEO {index}/{len(videos)}]"
        )

        # ----------------------------------------------------
        # Find annotation
        # ----------------------------------------------------

        annotation_path = find_annotation(
            video_path
        )

        if annotation_path:

            print(
                f"Matched annotation:\n"
                f"{annotation_path}"
            )

        else:

            print(
                "[WARNING] No matching annotation found."
            )

        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        try:

            rows = process_video(

                video_path,

                annotation_path,

                detector,

                save_frames=True
            )

            if rows:

                df_video = pd.DataFrame(
                    rows
                )

                # ------------------------------------------------
                # Temporal features
                # ------------------------------------------------

                df_video = add_temporal_features(

                    df_video,

                    window_size=10
                )

                all_rows.extend(

                    df_video.to_dict(

                        orient="records"
                    )
                )

                processed_videos += 1

            else:

                print(
                    "[WARNING] No usable pose rows."
                )

                failed_videos += 1

        except Exception as error:

            print(
                f"[ERROR] Video processing failed:"
                f"\n{error}"
            )

            failed_videos += 1

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    if not all_rows:

        raise RuntimeError(
            "No training features were extracted."
        )

    df = pd.DataFrame(
        all_rows
    )

    # --------------------------------------------------------
    # Clean numerical values
    # --------------------------------------------------------

    df = df.replace(

        [np.inf, -np.inf],

        np.nan
    )

    df = df.dropna(

        subset=
        FEATURE_COLUMNS
        +
        [
            "label"
        ]
    )

    # --------------------------------------------------------
    # Sort CSV
    # --------------------------------------------------------

    df = df.sort_values(

        [
            "dataset_folder",
            "video_name",
            "frame"
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save CSV INSIDE model folder
    # --------------------------------------------------------

    df.to_csv(

        FEATURE_CSV,

        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 75)

    print(
        f"Processed videos : "
        f"{processed_videos}"
    )

    print(
        f"Failed videos    : "
        f"{failed_videos}"
    )

    print(
        f"Training rows     : "
        f"{len(df)}"
    )

    print(
        f"FALL rows         : "
        f"{int((df['label'] == 1).sum())}"
    )

    print(
        f"NOT_FALL rows     : "
        f"{int((df['label'] == 0).sum())}"
    )

    print(
        f"Unique videos     : "
        f"{df['video'].nunique()}"
    )

    print(
        f"Unique video IDs  : "
        f"{df['video_name'].nunique()}"
    )

    print()

    print(
        f"CSV SAVED:\n{FEATURE_CSV}"
    )

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(df):

    print()
    print("=" * 75)
    print("TRAINING RANDOM FOREST FROM CSV")
    print("=" * 75)

    # --------------------------------------------------------
    # Verify features
    # --------------------------------------------------------

    missing_columns = [

        column

        for column in FEATURE_COLUMNS

        if column not in df.columns
    ]

    if missing_columns:

        raise RuntimeError(

            "Missing feature columns:\n"
            +
            "\n".join(missing_columns)
        )

    # --------------------------------------------------------
    # X / y / groups
    # --------------------------------------------------------

    X = df[
        FEATURE_COLUMNS
    ].copy()

    y = df[
        "label"
    ].astype(int)

    groups = df[
        "video"
    ]

    print(
        f"Total rows : {len(df)}"
    )

    print(
        f"Features   : {len(FEATURE_COLUMNS)}"
    )

    print(
        f"Videos     : {groups.nunique()}"
    )

    print()

    print(
        "Class distribution:"
    )

    print(

        y.value_counts()

        .rename(

            index={

                0: "NOT_FALL",

                1: "FALL"
            }
        )
    )

    # --------------------------------------------------------
    # Need enough groups
    # --------------------------------------------------------

    if groups.nunique() < 2:

        raise RuntimeError(

            "At least 2 unique videos are required "
            "for grouped train/test evaluation."
        )

    # --------------------------------------------------------
    # Group split
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(

        n_splits=1,

        test_size=0.20,

        random_state=42
    )

    train_idx, test_idx = next(

        splitter.split(

            X,

            y,

            groups=groups
        )
    )

    X_train = X.iloc[
        train_idx
    ]

    y_train = y.iloc[
        train_idx
    ]

    X_test = X.iloc[
        test_idx
    ]

    y_test = y.iloc[
        test_idx
    ]

    groups_train = groups.iloc[
        train_idx
    ]

    groups_test = groups.iloc[
        test_idx
    ]

    print()

    print(
        f"Training rows : "
        f"{len(X_train)}"
    )

    print(
        f"Testing rows  : "
        f"{len(X_test)}"
    )

    print(
        f"Training videos: "
        f"{groups_train.nunique()}"
    )

    print(
        f"Testing videos : "
        f"{groups_test.nunique()}"
    )

    print()

    print(
        "TEST VIDEOS:"
    )

    for video in groups_test.unique():

        print(
            "  ",
            video
        )

    # ========================================================
    # SMOTE
    # ========================================================

    try:

        from imblearn.over_sampling import SMOTE

        print()
        print(
            "Applying SMOTE to TRAINING DATA..."
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # SMOTE happens AFTER group splitting.
        # ----------------------------------------------------

        smote = SMOTE(

            random_state=42
        )

        X_train_resampled, y_train_resampled = (

            smote.fit_resample(

                X_train,

                y_train
            )
        )

        print(
            f"Before SMOTE: "
            f"{len(X_train)}"
        )

        print(
            f"After SMOTE : "
            f"{len(X_train_resampled)}"
        )

    except ImportError:

        print()

        print(
            "[WARNING] imbalanced-learn not installed."
        )

        print(
            "Training without SMOTE."
        )

        X_train_resampled = X_train

        y_train_resampled = y_train

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    model = RandomForestClassifier(

        n_estimators=500,

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        max_features="sqrt",

        class_weight="balanced_subsample",

        random_state=42,

        n_jobs=-1
    )

    print()
    print(
        "Training Random Forest..."
    )

    model.fit(

        X_train_resampled,

        y_train_resampled
    )

    print(
        "Training complete."
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(

        y_test,

        predictions
    )

    print()
    print("=" * 75)
    print("MODEL EVALUATION")
    print("=" * 75)

    print()

    print(
        f"ACCURACY: "
        f"{accuracy * 100:.2f}%"
    )

    print()

    print(
        "CLASSIFICATION REPORT:"
    )

    print(

        classification_report(

            y_test,

            predictions,

            target_names=[

                "NOT_FALL",

                "FALL"
            ],

            zero_division=0
        )
    )

    print(
        "CONFUSION MATRIX:"
    )

    print(

        confusion_matrix(

            y_test,

            predictions
        )
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    print()
    print(
        "FEATURE IMPORTANCE:"
    )

    importance = sorted(

        zip(

            FEATURE_COLUMNS,

            model.feature_importances_
        ),

        key=lambda x: x[1],

        reverse=True
    )

    for name, value in importance:

        print(

            f"{name:25s}"
            f"{value:.4f}"
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_package = {

        "model":
            model,

        "feature_columns":
            FEATURE_COLUMNS,

        "labels": {

            0: "NOT_FALL",

            1: "FALL"
        },

        "accuracy":
            float(accuracy),

        "mediapipe_features":
            BASE_FEATURE_COLUMNS,

        "temporal_features":
            TEMPORAL_FEATURE_COLUMNS,

        "window_size":
            10
    }

    joblib.dump(

        model_package,

        MODEL_PATH
    )

    print()
    print("=" * 75)

    print(
        f"MODEL SAVED:\n"
        f"{MODEL_PATH}"
    )

    print(
        f"\nModel size: "
        f"{os.path.getsize(MODEL_PATH) / (1024 * 1024):.2f} MB"
    )

    return model_package


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("SAFEFALL AI")
    print("FULL ANNOTATION-DRIVEN TRAINING PIPELINE")
    print("=" * 75)

    # --------------------------------------------------------
    # STEP 1
    # Check dataset
    # --------------------------------------------------------

    check_dataset()

    # --------------------------------------------------------
    # STEP 2
    # DELETE OLD CSV
    #
    # This guarantees that the new model is generated from
    # the newly extracted features.
    # --------------------------------------------------------

    if os.path.exists(
        FEATURE_CSV
    ):

        print()
        print(
            "Deleting old training_features.csv..."
        )

        os.remove(
            FEATURE_CSV
        )

    # --------------------------------------------------------
    # STEP 3
    # DELETE OLD EXTRACTED FRAMES
    #
    # Prevents old duplicate video folders from remaining.
    # --------------------------------------------------------

    if os.path.isdir(
        EXTRACTED_FRAMES_DIR
    ):

        print()
        print(
            "Removing old extracted frames..."
        )

        import shutil

        shutil.rmtree(
            EXTRACTED_FRAMES_DIR
        )

    os.makedirs(
        EXTRACTED_FRAMES_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # STEP 4
    # Create MediaPipe
    # --------------------------------------------------------

    detector = create_pose_detector()

    try:

        # ----------------------------------------------------
        # STEP 5
        # Extract features
        # ----------------------------------------------------

        df = extract_all_features(
            detector
        )

    finally:

        detector.close()

    # --------------------------------------------------------
    # STEP 6
    # Verify CSV
    # --------------------------------------------------------

    if not os.path.isfile(
        FEATURE_CSV
    ):

        raise RuntimeError(
            "\nCSV generation failed."
        )

    print()
    print(
        "CSV verification successful."
    )

    # --------------------------------------------------------
    # STEP 7
    # IMPORTANT:
    # Reload CSV from disk.
    #
    # This means the model is genuinely generated from the
    # CSV file that was written to model/.
    # --------------------------------------------------------

    print()
    print(
        "Reloading training CSV..."
    )

    df_from_csv = pd.read_csv(
        FEATURE_CSV
    )

    print(
        f"Rows loaded from CSV: "
        f"{len(df_from_csv)}"
    )

    # --------------------------------------------------------
    # STEP 8
    # Generate PKL from CSV
    # --------------------------------------------------------

    train_model(
        df_from_csv
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("SAFEFALL PIPELINE COMPLETE")
    print("=" * 75)

    print()
    print(
        "FILES GENERATED:"
    )

    print()

    print(
        f"CSV:\n{FEATURE_CSV}"
    )

    print()

    print(
        f"MODEL:\n{MODEL_PATH}"
    )

    print()

    print(
        f"EXTRACTED FRAMES:\n"
        f"{EXTRACTED_FRAMES_DIR}"
    )

    print()
    print(
        "Every video now has a unique ID based on:"
    )

    print(
        "DATASET_FOLDER + VIDEO_NAME"
    )

    print()
    print(
        "Example:"
    )

    print(
        "Coffee_room_01__video (1)"
    )

    print(
        "Home_01__video (1)"
    )

    print()
    print(
        "These are treated as DIFFERENT videos."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
