# Posture Analysis with Pose and FaceMesh Landmarks

This Python script utilizes the Mediapipe library, OpenCV, and NumPy to perform real-time posture analysis using landmarks detected from both body pose and facial features. The script captures video from a webcam, identifies key body and facial landmarks, and calculates various posture metrics.

## Features:
- Body pose analysis using `mp_pose` from Mediapipe.
- Facial landmarks detection using `mp_face_mesh` from Mediapipe.
- Calculation of angles and distances between specific body landmarks for posture assessment.
- Visualization of body pose landmarks, connections, and posture markers on the video feed.
- Display of real-time metrics such as trunk angle, head angle, shoulder relaxation, and overall straightness.

## Requirements:
- Python 3.x
- OpenCV (`cv2`)
- Mediapipe (`mediapipe`)
- NumPy (`numpy`)

## Usage:
1. Install the required libraries: `pip install opencv-python mediapipe numpy`
2. Run the script: `python script_name.py`
3. Adjust the webcam if needed and observe the real-time posture analysis.

## Notes:
- The script automatically chooses the correct video source (0 or 1) based on the availability of the default source.
- Posture markers are displayed on the video feed with color-coded indicators for easy interpretation.
- Press 'q' to exit the script.

Feel free to customize the script or integrate it into your applications for real-time posture monitoring.
