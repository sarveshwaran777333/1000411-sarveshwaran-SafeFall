# 1000411-sarveshwaran-SafeFall

# 🛡️ SafeFall AI

## AI-Based Fall Detection System

SafeFall AI is an Artificial Intelligence-based fall detection system designed to identify potential falls from video footage.

The system uses **MediaPipe Pose Landmarker** to extract human body-pose information from video frames and a trained **Random Forest Classifier** to determine whether the observed movement is a fall or a normal movement.

The application provides the detection result through an interactive **Streamlit dashboard**, along with fall-risk probabilities, evidence frames, timestamps, and an analysis summary.

---

## 📌 Project Overview

Falls can be dangerous, particularly for elderly people, and detecting them quickly can help enable a faster response.

SafeFall AI analyzes video footage frame by frame and extracts information about the person's posture and movement. These features are then passed to a trained machine-learning model.

The system produces:

- Fall / No Fall classification
- Fall-risk probability
- Highest-risk frame
- Highest-risk timestamp
- Number of high-risk frames
- Evidence frames
- Detected fall region
- Fall duration
- Fall-risk timeline
- Detailed frame predictions
- CSV results for further analysis

---

## 🎯 Project Objective

The main objective of SafeFall AI is to develop a computer-vision-based system capable of detecting falls automatically from video footage.

The project focuses on:

1. Detecting human body posture using MediaPipe Pose.
2. Extracting numerical posture and movement features.
3. Using temporal information to understand movement across multiple frames.
4. Classifying movements using a Random Forest machine-learning model.
5. Confirming a fall using multiple high-risk frames rather than relying on a single frame.
6. Presenting the results through an easy-to-use Streamlit dashboard.

---

## 🧠 AI and Machine Learning Approach

SafeFall AI uses a combination of computer vision and machine learning.

### 1. MediaPipe Pose

MediaPipe Pose Landmarker is used to identify human body landmarks from video frames.

The extracted pose information is converted into numerical features that describe the person's posture and movement.

### 2. Feature Engineering

The model uses **13 features** in total.

These features include information related to:

- Body aspect ratio
- Torso angle
- Hip position
- Torso ratio
- Bounding-box width
- Bounding-box height
- Hip vertical velocity
- Torso-angle velocity
- Rolling maximum hip velocity
- Rolling mean torso angle
- Rolling maximum aspect ratio
- Rolling minimum hip position
- Torso-angle change range

The temporal features allow the model to consider movement over multiple frames instead of analyzing each frame independently.

### 3. Temporal Analysis

The system uses a **10-frame temporal window**.

This means that movement information from a sequence of frames is considered when generating the prediction.

Temporal information helps distinguish sudden changes in posture from normal body movement.

---

## 🌲 Machine Learning Model

SafeFall AI uses a **Random Forest Classifier**.

The trained model contains:

| Parameter | Value |
|---|---|
| Model | Random Forest Classifier |
| Number of trees | 500 |
| Class weighting | Balanced subsample |
| Random state | 42 |
| Number of features | 13 |
| Temporal window | 10 frames |
| Classes | FALL, NOT_FALL |

The trained model is stored in:

```text
model/safe_fall_model.pkl
