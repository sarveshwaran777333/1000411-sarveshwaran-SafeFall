print("SafeFall AI - TRAIN MODEL (FFmpeg Native)")
print("=" * 70)

import os

# Suppress MediaPipe internal C++ logging
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import glob
import re
import math
import urllib.request
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
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_ROOT = r"D:\AI\ib_ai(a)\year 2\archive"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "safe_fall_model.pkl")
FEATURE_CSV = os.path.join(PROJECT_ROOT, "training_features.csv")
POSE_MODEL_PATH = os.path.join(PROJECT_ROOT, "pose_landmarker_full.task")

POSE_MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/"
    "pose_landmarker/"
    "pose_landmarker_full/"
    "float16/1/"
    "pose_landmarker_full.task"
)


# ============================================================
# TRAINING FOLDERS
# ============================================================

TRAINING_FOLDERS = [
    "Coffee_room_01",
    "Coffee_room_02",
    "Home_01",
    "Home_02"
]


# ============================================================
# MEDIA / WARNING CLEANUP
# ============================================================

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset():
    print()
    print("=" * 70)
    print("[1/7] CHECKING LOCAL DATASET")
    print("=" * 70)

    print("\nDataset location:", DATASET_ROOT)

    if not os.path.isdir(DATASET_ROOT):
        raise FileNotFoundError(
            f"\nDataset folder was not found:\n{DATASET_ROOT}"
        )

    print("\n[OK] Dataset root exists.\nChecking training folders...")

    found_count = 0
    for folder in TRAINING_FOLDERS:
        folder_path = os.path.join(DATASET_ROOT, folder)
        if os.path.isdir(folder_path):
            print("[FOUND]  ", folder)
            found_count += 1
        else:
            print("[MISSING]", folder)

    if found_count == 0:
        raise FileNotFoundError(
            "\nNone of the expected training folders were found."
        )

    print(f"\n[OK] Training folders found: {found_count} / {len(TRAINING_FOLDERS)}")
    return DATASET_ROOT


# ============================================================
# DOWNLOAD MEDIAPIPE MODEL
# ============================================================

def prepare_pose_model():
    print()
    print("=" * 70)
    print("[2/7] PREPARING MEDIAPIPE POSE MODEL")
    print("=" * 70)

    if os.path.exists(POSE_MODEL_PATH):
        print("\n[OK] Pose Landmarker already exists.")
        print("Location:", POSE_MODEL_PATH)
        return

    print("\n[*] Pose Landmarker model not found.")
    print("[*] Downloading MediaPipe Pose Landmarker...")

    try:
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    except Exception as e:
        if os.path.exists(POSE_MODEL_PATH):
            os.remove(POSE_MODEL_PATH)
        raise RuntimeError(
            f"\nCould not download MediaPipe Pose Landmarker.\nError: {e}"
        )

    print("\n[OK] Pose Landmarker downloaded.")
    print("Location:", POSE_MODEL_PATH)


# ============================================================
# CREATE MEDIAPIPE DETECTOR
# ============================================================

def create_pose_detector():
    print()
    print("=" * 70)
    print("[3/7] INITIALIZING MEDIAPIPE")
    print("=" * 70)

    try:
        base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.25,
            min_pose_presence_confidence=0.25,
            min_tracking_confidence=0.25
        )
        detector = vision.PoseLandmarker.create_from_options(options)
        print("\n[OK] MediaPipe Pose Landmarker initialized.")
        return detector
    except Exception as e:
        raise RuntimeError(f"\nCould not initialize MediaPipe.\nError: {e}")


# ============================================================
# FIND ALL TRAINING VIDEOS
# ============================================================

def find_training_videos(dataset_root):
    print()
    print("=" * 70)
    print("[4/7] FINDING TRAINING VIDEOS")
    print("=" * 70)

    videos = []
    for folder in TRAINING_FOLDERS:
        folder_path = os.path.join(dataset_root, folder)
        if not os.path.isdir(folder_path):
            print("[WARNING] Missing:", folder)
            continue

        found = glob.glob(
            os.path.join(folder_path, "**", "*.avi"),
            recursive=True
        )
        print(f"[FOUND] {folder} : {len(found)} AVI videos")
        videos.extend(found)

    unique_videos = []
    seen = set()

    for video in videos:
        normalized = os.path.normcase(os.path.abspath(video))
        if normalized not in seen:
            seen.add(normalized)
            unique_videos.append(video)

    videos = sorted(unique_videos)
    print(f"\n[OK] Total unique training videos: {len(videos)}")

    if len(videos) == 0:
        raise RuntimeError("No AVI videos found in the training folders.")

    return videos


# ============================================================
# FIND ANNOTATION
# ============================================================

def find_annotation(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    search_root = os.path.dirname(video_path)

    candidates = glob.glob(
        os.path.join(search_root, "**", video_name + ".txt"),
        recursive=True
    )
    if candidates:
        return sorted(candidates)[0]

    candidates = glob.glob(
        os.path.join(DATASET_ROOT, "**", video_name + ".txt"),
        recursive=True
    )
    if candidates:
        return sorted(candidates)[0]

    return None


# ============================================================
# READ ANNOTATION
# ============================================================

def read_annotation(annotation_path):
    try:
        with open(annotation_path, "r", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print("[ERROR] Could not read annotation:", annotation_path)
        print("Reason:", e)
        return None, None

    standalone_numbers = []
    for index, line in enumerate(lines):
        if re.fullmatch(r"\d+", line):
            standalone_numbers.append((index, int(line)))

    if len(standalone_numbers) < 2:
        return None, None

    if standalone_numbers[0][0] == 0:
        fall_start = standalone_numbers[0][1]
        fall_end = standalone_numbers[1][1]
        return fall_start, fall_end

    fall_start = standalone_numbers[-2][1]
    fall_end = standalone_numbers[-1][1]
    return fall_start, fall_end


# ============================================================
# FFPROBE METADATA EXTRACTION
# ============================================================

def get_video_info(video_path):
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
        "-of", "csv=p=0",
        "-count_frames",
        video_path
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        parts = result.stdout.strip().split(",")
        width = int(parts[0])
        height = int(parts[1])
        
        # Parse frame rate fraction (e.g. "30/1" or "29.97")
        num, den = parts[2].split("/") if "/" in parts[2] else (parts[2], 1)
        fps = float(num) / float(den) if float(den) > 0 else 30.0
        
        frame_count = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
        return width, height, fps, frame_count
    except Exception:
        return None, None, 30.0, 0


# ============================================================
# CALCULATE POSE FEATURES
# ============================================================

def calculate_features(landmarks, previous_landmarks=None):
    if landmarks is None or len(landmarks) < 25:
        return None

    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    lh = landmarks[LEFT_HIP]
    rh = landmarks[RIGHT_HIP]

    visibility = [
        getattr(ls, "visibility", 1.0),
        getattr(rs, "visibility", 1.0),
        getattr(lh, "visibility", 1.0),
        getattr(rh, "visibility", 1.0)
    ]

    if min(visibility) < 0.15:
        return None

    shoulder_x = (ls.x + rs.x) / 2.0
    shoulder_y = (ls.y + rs.y) / 2.0
    hip_x = (lh.x + rh.x) / 2.0
    hip_y = (lh.y + rh.y) / 2.0

    visible_points = [
        lm for lm in landmarks 
        if getattr(lm, "visibility", 1.0) > 0.15
    ]

    if len(visible_points) < 4:
        return None

    x_values = [lm.x for lm in visible_points]
    y_values = [lm.y for lm in visible_points]

    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    bbox_w = max(0.01, max_x - min_x)
    bbox_h = max(0.01, max_y - min_y)
    aspect_ratio = bbox_w / bbox_h

    dx = shoulder_x - hip_x
    dy = shoulder_y - hip_y

    torso_angle = math.degrees(math.atan2(abs(dy), abs(dx) + 1e-8))
    torso_angle = float(np.clip(torso_angle, 0.0, 90.0))

    hip_y_norm = float(np.clip(hip_y, 0.0, 1.0))
    torso_length = math.sqrt(dx ** 2 + dy ** 2)
    torso_ratio = torso_length / bbox_h

    v_hip_y = 0.0
    v_torso_angle = 0.0

    if previous_landmarks is not None and len(previous_landmarks) >= 25:
        prev_ls = previous_landmarks[11]
        prev_rs = previous_landmarks[12]
        prev_lh = previous_landmarks[23]
        prev_rh = previous_landmarks[24]

        prev_hip_y = (prev_lh.y + prev_rh.y) / 2.0
        v_hip_y = hip_y_norm - prev_hip_y

        prev_shoulder_x = (prev_ls.x + prev_rs.x) / 2.0
        prev_shoulder_y = (prev_ls.y + prev_rs.y) / 2.0
        prev_hip_x = (prev_lh.x + prev_rh.x) / 2.0
        prev_hip_y_center = (prev_lh.y + prev_rh.y) / 2.0

        prev_dx = prev_shoulder_x - prev_hip_x
        prev_dy = prev_shoulder_y - prev_hip_y_center

        prev_angle = math.degrees(math.atan2(abs(prev_dy), abs(prev_dx) + 1e-8))
        prev_angle = float(np.clip(prev_angle, 0.0, 90.0))

        v_torso_angle = torso_angle - prev_angle

    return {
        "aspect_ratio": float(aspect_ratio),
        "torso_angle": float(torso_angle),
        "hip_y": float(hip_y_norm),
        "torso_ratio": float(torso_ratio),
        "bbox_w": float(bbox_w),
        "bbox_h": float(bbox_h),
        "v_hip_y": float(v_hip_y),
        "v_torso_angle": float(v_torso_angle)
    }


# ============================================================
# PROCESS ONE VIDEO VIA FFMPEG PIPE (NO OPENCV)
# ============================================================

def process_video(video_path, annotation_path, detector):
    fall_start, fall_end = read_annotation(annotation_path)
    if fall_start is None:
        return [], 0, 0

    width, height, fps, frame_count = get_video_info(video_path)
    if not width or not height:
        print("   -> Could not extract video parameters with ffprobe")
        return [], 0, 0

    if fall_start != 0 or fall_end != 0:
        fall_start = max(1, fall_start)
        if frame_count > 0:
            fall_end = min(fall_end, frame_count)

    command = [
        "ffmpeg",
        "-loglevel", "quiet",
        "-an",                      # Ignore audio completely
        "-i", video_path,
        "-f", "image2pipe",
        "-pix_fmt", "rgb24",        # Output raw RGB24 frames
        "-vcodec", "rawvideo",
        "-"
    ]

    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)

    features = []
    previous_landmarks = None
    frame_number = 0
    pose_count = 0
    decoded_count = 0
    last_timestamp_ms = -1
    frame_bytes_size = width * height * 3

    while True:
        raw_frame = pipe.stdout.read(frame_bytes_size)
        if len(raw_frame) != frame_bytes_size:
            break

        frame_number += 1
        decoded_count += 1

        frame_array = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_array
        )

        timestamp_ms = int(((frame_number - 1) / fps) * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        try:
            result = detector.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            previous_landmarks = None
            continue

        if result.pose_landmarks is None or len(result.pose_landmarks) == 0:
            previous_landmarks = None
            continue

        landmarks = result.pose_landmarks[0]
        pose_features = calculate_features(landmarks, previous_landmarks)

        if pose_features is None:
            previous_landmarks = landmarks
            continue

        pose_count += 1

        if fall_start == 0 and fall_end == 0:
            label = 0
        elif fall_start <= frame_number <= fall_end:
            label = 1
        else:
            label = 0

        pose_features["label"] = label
        pose_features["video"] = os.path.abspath(video_path)
        pose_features["video_name"] = os.path.basename(video_path)
        pose_features["frame"] = frame_number

        features.append(pose_features)
        previous_landmarks = landmarks

    pipe.stdout.close()
    pipe.wait()

    return features, decoded_count, pose_count


# ============================================================
# EXTRACT ALL FEATURES
# ============================================================

def extract_all_features(videos, dataset_root, detector):
    print()
    print("=" * 70)
    print("[5/7] EXTRACTING POSE FEATURES")
    print("=" * 70)

    all_rows = []
    successful_videos = 0
    failed_videos = 0
    total_decoded = 0
    total_pose = 0
    total_fall = 0
    total_not_fall = 0

    for index, video_path in enumerate(videos, 1):
        relative_path = os.path.relpath(video_path, dataset_root)
        print(f"\n[{index}/{len(videos)}] {relative_path}")

        annotation_path = find_annotation(video_path)
        if annotation_path is None:
            print("   -> ANNOTATION NOT FOUND")
            failed_videos += 1
            continue

        fall_start, fall_end = read_annotation(annotation_path)
        if fall_start is None:
            print("   -> INVALID ANNOTATION")
            failed_videos += 1
            continue

        print(f"   -> Annotation: {fall_start} -> {fall_end}")

        rows, decoded_count, pose_count = process_video(
            video_path, annotation_path, detector
        )

        total_decoded += decoded_count
        total_pose += pose_count

        print("   -> Total decoded frames:", decoded_count)
        print("   -> Pose frames:", pose_count)

        if len(rows) == 0:
            print("   -> No pose features extracted")
            failed_videos += 1
            continue

        all_rows.extend(rows)
        successful_videos += 1

        fall_count = sum(row["label"] == 1 for row in rows)
        not_fall_count = sum(row["label"] == 0 for row in rows)

        total_fall += fall_count
        total_not_fall += not_fall_count

        print("   -> FALL:", fall_count)
        print("   -> NOT_FALL:", not_fall_count)

    print()
    print("=" * 70)
    print("POSE EXTRACTION COMPLETE")
    print("=" * 70)
    print("Successful videos:", successful_videos)
    print("Failed videos:", failed_videos)
    print("Total decoded frames:", total_decoded)
    print("Total pose frames:", total_pose)
    print("FALL pose rows:", total_fall)
    print("NOT_FALL pose rows:", total_not_fall)

    return all_rows, successful_videos, failed_videos


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_model(df):
    print()
    print("=" * 70)
    print("[6/7] TRAINING RANDOM FOREST")
    print("=" * 70)

    feature_columns = [
        "aspect_ratio",
        "torso_angle",
        "hip_y",
        "torso_ratio",
        "bbox_w",
        "bbox_h",
        "v_hip_y",
        "v_torso_angle"
    ]

    X = df[feature_columns].astype(float)
    y = df["label"].astype(int)
    groups = df["video"]

    print("\nTotal feature rows:", len(df))
    print("NOT_FALL rows:", int((y == 0).sum()))
    print("FALL rows:", int((y == 1).sum()))

    if len(y.unique()) < 2:
        raise RuntimeError("Training data contains only one class.")

    df.to_csv(FEATURE_CSV, index=False)
    print("\n[OK] Training features saved:\n", FEATURE_CSV)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_indices, test_indices = next(splitter.split(X, y, groups))

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("\nTraining videos:", groups.iloc[train_indices].nunique())
    print("Testing videos:", groups.iloc[test_indices].nunique())

    if len(y_train.unique()) < 2:
        raise RuntimeError(
            "Training split contains only one class. "
            "The dataset split cannot train both classes."
        )

    if len(y_test.unique()) < 2:
        print("\n[WARNING] Test split contains only one class.")

    print("\n[*] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print()
    print("=" * 70)
    print("MODEL ACCURACY:", f"{accuracy * 100:.2f}%")
    print("=" * 70)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["NOT_FALL", "FALL"],
            zero_division=0
        )
    )

    print("\nFeature importance:")
    importance = sorted(
        zip(feature_columns, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    for name, value in importance:
        print(f"  {name:<20} {value:.4f}")

    print("\n[*] Saving trained model...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_package = {
        "model": model,
        "feature_columns": feature_columns,
        "labels": {0: "NOT_FALL", 1: "FALL"},
        "accuracy": float(accuracy),
        "mediapipe_features": feature_columns
    }

    joblib.dump(model_package, MODEL_PATH)

    print("\n[OK] Model saved:\n", MODEL_PATH)
    return accuracy, len(X_train), len(X_test)


# ============================================================
# MAIN EXECUTION PIPELINE
# ============================================================

def main():
    dataset_root = check_dataset()
    prepare_pose_model()
    detector = create_pose_detector()

    try:
        videos = find_training_videos(dataset_root)
        all_rows, successful_videos, failed_videos = extract_all_features(
            videos, dataset_root, detector
        )

        if not all_rows:
            raise RuntimeError("No feature records were extracted across all dataset videos.")

        df = pd.DataFrame(all_rows)
        
        print()
        print("=" * 70)
        print("[7/7] EXECUTING TRAINING")
        print("=" * 70)
        
        train_model(df)

    finally:
        detector.close()
        print("\n[OK] Pipeline completed successfully.")


if __name__ == "__main__":
    main()
