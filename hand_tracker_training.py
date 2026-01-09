import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open CSV and add a header if it's new (Optional but helpful)
with open('hand_data.csv', mode='a', newline='') as f:
    writer = csv.writer(f)

    cap = cv2.VideoCapture(0)
    label = "HELLO"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # zip combines landmarks and handedness so we process them together
            for idx, (hand_lms, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                
                # 1. Get Hand Side and Finger Info
                hand_side = handedness.classification[0].label # "Left" or "Right"

                tip_ids = [4, 8, 12, 16, 20]
                fingers = []
                
                if hand_side == "Right":
                    if hand_lms.landmark[tip_ids[0]].x < hand_lms.landmark[tip_ids[0]-1].x:
                        fingers.append(1) 
                    else:
                        fingers.append(0)
                else:
                    if hand_lms.landmark[tip_ids[0]].x > hand_lms.landmark[tip_ids[0]-1].x:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                for id in range(1, 5):
                    if hand_lms.landmark[tip_ids[id]].y < hand_lms.landmark[tip_ids[id] - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                total_fingers = fingers.count(1)

                # 2. Draw Landmarks and Text
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f"{hand_side}: {fingers} Count: {total_fingers}", 
                            (50, 50 + (idx*50)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # 3. Extract all 63 coordinates (21 pts * x,y,z)
                landmarks = []
                for lm in hand_lms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # 4. Save to CSV when 's' is pressed
                # We save: [Label, Side, x1, y1, z1, x2, y2, z2...]
                key = cv2.waitKey(1)
                if key & 0xFF == ord('s'):
                    writer.writerow([label, hand_side] + landmarks)
                    print(f"Saved {label} sample. Fingers up: {total_fingers}")

        cv2.imshow("Collecting Data", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()