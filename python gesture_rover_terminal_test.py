import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_gesture(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = [0,0,0,0,0]

    # Thumb check
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0]-1].x:
        fingers[0] = 1
    # Other fingers
    for i in range(1,5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i]-2].y:
            fingers[i] = 1

    # Optional: Use numpy for extra calculations if needed
    points = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
    center = np.mean(points, axis=0)

    if fingers == [0,1,0,0,0]: return "forward"    # Index finger
    elif fingers == [1,0,0,0,0]: return "right"    # Thumb
    elif fingers == [0,0,0,0,1]: return "left"     # Little finger
    elif fingers == [0,1,1,0,0]: return "reverse"  # Index + Middle
    else: return "stop"

# 📸 Choose one:
# 1) Laptop webcam:
# cap = cv2.VideoCapture(0)

# 2) PHONE as webcam: Replace the URL below with your phone IP Webcam URL (shown in the app)
cap = cv2.VideoCapture("http://192.168.1.100:8080/video")  # <-- Replace this with your phone's URL

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Check webcam or phone IP.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7).process(rgb)

    cmd = "stop"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cmd = detect_gesture(hand_landmarks)

    # 👉 Instead of sending to ESP32, just print to terminal
    print(f"Command: {cmd}")

    cv2.putText(frame, f"Command: {cmd}", (10,40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow("Gesture Control Test (Terminal Only)", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
