import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import time
import os

# --- 1. MODEL CHECK ---
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print(f"ERROR: '{model_path}' not found! Download it from Google's MediaPipe site.")
    exit()

# --- 2. Initialize MediaPipe Tasks Hand Landmarker ---
# In 2026, we avoid mp.solutions entirely to prevent AttributeError
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

with open('hand_data.csv', mode='a', newline='') as f:
    writer = csv.writer(f)
    cap = cv2.VideoCapture(0)
    label = "HELLO"

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # MediaPipe Tasks requires mp.Image and a timestamp for VIDEO mode
        # We flip the frame horizontally for a more natural "mirror" feel
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(time.time() * 1000)
        
        # This call is where the 'NoneType' error usually stems from if detector init failed
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for idx, (hand_landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
                
                # FIX: In Tasks API, handedness is a list of categories
                hand_side = handedness[0].category_name 

                # Finger Counting Logic
                tip_ids = [4, 8, 12, 16, 20]
                fingers = []
                
                # Thumb (logic flipped because we flipped the frame)
                if hand_side == "Left": # Left hand in mirror
                    fingers.append(1 if hand_landmarks[tip_ids[0]].x < hand_landmarks[tip_ids[0]-1].x else 0)
                else: # Right hand in mirror
                    fingers.append(1 if hand_landmarks[tip_ids[0]].x > hand_landmarks[tip_ids[0]-1].x else 0)

                # Other 4 fingers
                for i in range(1, 5):
                    fingers.append(1 if hand_landmarks[tip_ids[i]].y < hand_landmarks[tip_ids[i]-2].y else 0)
                
                total_fingers = fingers.count(1)

                # --- 3. Visualization (Manual Drawing) ---
                h, w, _ = frame.shape
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                cv2.putText(frame, f"{hand_side}: {total_fingers} Fingers", 
                            (20, 50 + (idx*40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # --- 4. Save to CSV ---
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    landmarks_flat = []
                    for lm in hand_landmarks:
                        landmarks_flat.extend([lm.x, lm.y, lm.z])
                    writer.writerow([label, hand_side] + landmarks_flat)
                    print(f"Saved {label} sample.")

        cv2.imshow("Hand Tracker 2026", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
detector.close()
