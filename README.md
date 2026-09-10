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
```

📊 Model Performance

The trained model achieved an accuracy of:

95.47%

This result represents the accuracy stored with the trained model package.

The system also uses fall-risk probabilities and temporal confirmation logic during video inference rather than treating one isolated prediction as a confirmed fall.

📂 Dataset

The model was trained using the Le2i Fall Detection Dataset.

The dataset contains video recordings representing fall and non-fall activities.

The videos were processed to extract frames and human-pose information. The extracted features were then used to create the machine-learning training dataset.

The processed training data contained:

39,797 feature rows
130 unique videos
37,457 NOT_FALL samples
2,340 FALL samples

The data was processed using MediaPipe pose estimation before being used for machine-learning training.

🔄 Data Processing Pipeline

The overall pipeline used by SafeFall AI is:

Input Video
     ↓
Video Frame Extraction
     ↓
MediaPipe Pose Detection
     ↓
Human Body Landmarks
     ↓
Feature Extraction
     ↓
Temporal Feature Calculation
     ↓
Random Forest Classification
     ↓
Fall-Risk Probability
     ↓
Temporal Fall Confirmation
     ↓
Streamlit Results Dashboard
🧪 Fall Confirmation Logic

The system does not immediately classify a video as a fall based only on one prediction.

Instead, it uses configurable detection settings.

Fall Probability Threshold

The user can configure the probability threshold between:

0.40 – 0.90

The default threshold is:

0.50
Minimum High-Risk Frames

The application can require multiple high-risk frames before confirming a fall.

The configurable range is:

1 – 6 frames

The default is:

2 frames
Maximum Frame Gap

The application also allows a configurable gap between high-risk frames when grouping them into a fall event.

The available range is:

1 – 5 frames

The default is:

2 frames

This approach helps reduce the chance of treating an isolated high-risk prediction as a confirmed fall.

🖥️ Streamlit Application

SafeFall AI is implemented as an interactive Streamlit web application.

The application allows users to upload video files and analyze them using the trained AI model.

Supported Video Formats
.mp4
.avi
.mov
.mkv
.mpeg
.mpg

After processing, the dashboard displays the detection results and supporting evidence.

📈 Dashboard Features

The SafeFall AI dashboard provides:

Detection Result

The system displays either:

FALL DETECTED

or

NO FALL DETECTED
Risk Metrics

The application displays:

Maximum fall risk
Number of high-risk frames
Number of pose frames analyzed
Highest-risk timestamp
Fall Evidence

When a fall is detected, the dashboard provides:

Evidence start time
Evidence end time
Evidence frames
Detected fall region
Event duration
Event timeline
Risk Visualization

The application provides a fall-risk chart showing how the predicted fall probability changes throughout the video.

Detailed Predictions

Frame-level predictions can also be inspected to understand how the model behaved throughout the video.

CSV Export

The application provides a downloadable CSV file containing detection results.

The output file is:

safefall_detection_results.csv
🛠️ Technologies Used
Programming Language
Python 3.13
Computer Vision
OpenCV
MediaPipe Pose Landmarker
Machine Learning
Scikit-learn
Random Forest Classifier
Joblib
Web Application
Streamlit
Data Processing
NumPy
Pandas
📦 Project Structure
SafeFall/
│
├── app.py
│
├── model/
│   └── safe_fall_model.pkl
│
├── pose_landmarker_full.task
│
├── requirements.txt
│
└── README.md
⚙️ Installation

Clone the repository:

git clone https://github.com/sarveshwaran777333/1000411-sarveshwaran-SafeFall.git

Move into the project folder:

cd 1000411-sarveshwaran-SafeFall

Install the required packages:

pip install -r requirements.txt
▶️ Running the Application

Run the Streamlit application using:

streamlit run app.py

The application will open in a browser.

Upload a supported video file and configure the detection settings if required.

The system will then process the video and display the AI-based fall detection results.

🔬 Testing

The system can be tested using different types of video conditions, including:

Normal activities
Fall movements
Different body positions
Different camera viewpoints
Different backgrounds
Different movement speeds

Testing helps identify:

Missed detections
False positives
Incorrect classifications
Pose-detection failures

The results can be used to identify areas where the dataset, feature engineering, or model could be improved.

⚠️ Limitations

SafeFall AI has several limitations.

Camera Dependency

The accuracy of pose detection depends on the quality and position of the camera.

Occlusion

If important parts of the person's body are hidden, MediaPipe may not be able to detect the required landmarks correctly.

Lighting

Very poor lighting can reduce the quality of pose detection.

Unusual Movements

Some movements may resemble a fall and can potentially produce high fall-risk predictions.

Dataset Limitations

Machine-learning performance depends on the data used during training. Different environments, camera angles, and activities may produce different results.

Real-World Safety

SafeFall AI is an academic AI project and should not be treated as a replacement for professional emergency or medical monitoring systems.

🚀 Future Improvements

Possible future improvements include:

Increasing the size and diversity of the training dataset.
Adding more real-world video scenarios.
Improving pose detection under occlusion.
Testing additional machine-learning models.
Improving false-positive reduction.
Adding real-time CCTV support.
Adding multiple-person detection.
Adding notification or alert systems.
Optimizing the system for edge devices.
Improving performance on low-light video.
📚 Research and Learning

This project demonstrates the application of several Artificial Intelligence concepts:

Computer Vision
Pose Estimation
Feature Engineering
Supervised Machine Learning
Random Forest Classification
Temporal Feature Analysis
Probability-based Classification
Video Processing
Model Evaluation
Streamlit Deployment

The project shows how computer-vision data can be converted into numerical features and used to build a machine-learning system for an applied problem.

🌐 Deployment

The application is designed to run using Streamlit.

GitHub Repository:

https://github.com/sarveshwaran777333/1000411-sarveshwaran-SafeFall
👨‍💻 Project Information

Project: SafeFall AI
Title: AI-Based Fall Detection System
Course: Artificial Intelligence
Machine Learning Model: Random Forest Classifier
Computer Vision: MediaPipe Pose Landmarker
Application Framework: Streamlit
Model Accuracy: 95.47%

📌 Conclusion

SafeFall AI demonstrates how Artificial Intelligence, computer vision, and machine learning can be combined to create an automated fall-detection system.

By extracting human-pose information with MediaPipe and analyzing posture and movement using a Random Forest model, the system can estimate fall risk from video footage.

The Streamlit dashboard makes the system easier to use by providing the final classification together with risk metrics, evidence frames, timestamps, visualizations, and downloadable results.

The trained model achieved 95.47% accuracy, providing a strong foundation for further development and testing.


**One important thing:** don't write **“99% accuracy”** in the README. Your actual trained model reports **95.47%**, so use that number for the SA documentation.
