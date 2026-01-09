# Hand Gesture Data Collector

A Python-based utility for capturing 3D hand landmarks using MediaPipe and OpenCV. This tool is designed to streamline the creation of custom datasets for hand gesture recognition and machine learning models.

## Overview

The application processes live video feed to detect hand landmarks and calculate finger states. When triggered, it exports the raw spatial data (x, y, z coordinates) along with metadata to a structured CSV file.

## Key Features

* **Dual Hand Support**: Tracks and processes up to two hands simultaneously.
* **Handedness Classification**: Automatically labels data as "Left" or "Right" to account for mirroring.
* **Finger State Logic**: Real-time calculation of which fingers are extended or folded.
* **Flattened Data Export**: Converts 21 3D landmarks into a 63-feature row optimized for machine learning training.

## Requirements

### Hardware
* Webcam (Integrated or USB)
* ThinkPad T14 gen 4 (Ubuntu 25.10)
* Thinkpad P16 gen 2 (Windows 11, 25H2)

### Software
* Python 3.10+
* OpenCV
* Hand Landmarker
```
Google Hand Handmarker Download:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```
* MediaPipe

```
pip install opencv-python mediapipe
```

## Data structure
The script saves data to ```"hand_data.csv```. Each row follows this schema:


| Column  |Description|
| ------------- | ------------- |
| Label  | The string name of the gesture |
| Side | Handedness (Left of Right) |
| Landmarks 1--63 | 21 points with 3 coordinates (x, y, z) each |

## Usage

### 1. Configure label: 
Open the script and update the ```label``` variable to the gesture you wish to record

```label = "YOUR_GESTURE_NAME"```

### 2. Execute Script: 
``` hand_tracker_training.py ```

### 3. Controls:
- S Key: Save the current frame's landmarks to the CSV.
- Q Key: Safely exit the application and release camera resources.

## Technical Implementation
Technical Implementation
Finger Counting Algorithm
The script determines if a finger is "up" (1) or "down" (0) using the following logic:

Thumb: Checks if the Tip (ID 4) is outside the IP joint (ID 3) on the X-axis, adjusted for handedness.

Fingers: Checks if the Tip (IDs 8, 12, 16, 20) is above the PIP joint (ID 6, 10, 14, 18) on the Y-axis.

## License
This project is open-source and available under the MIT License.


