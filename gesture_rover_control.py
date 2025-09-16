
import cv2
import mediapipe as mp
import requests
import numpy as np
import time

# 👉 YAHAN apne ESP32 ka local IP address dalen
ESP32_IP = "192.168.1.45"  # e.g., 192.168.1.45 (replace with your ESP32 IP)

# 👉 YAHAN apne PHONE ka IP:PORT dalen jo IP Webcam app dikhata hai
# Example: "http://192.168.1.100:8080/video"
PHONE_CAM_URL = "http://192.168.1.100:8080/video"  # <-- Replace with your phone's IP webcam URL

last_cmd = ""
last_time = 0

def send_command(cmd):
    global last_cmd, last_time
    now = time.time()
    if cmd != last_cmd or now - last_time > 0.5:
        try:
            url = f"http://{ESP32_IP}/{cmd}"
            r = requests.get(url, timeout=0.5)
            print(f"Sent: {cmd}, Response: {r.status_code}")
        except Exception as e:
            print(f"Failed to send {cmd}: {e}")
        last_cmd, last_time = cmd, now

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

    # Optional: Use numpy for future calculations (e.g., hand center)
    points = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
    center = np.mean(points, axis=0)

    if fingers == [0,1,0,0,0]: return "forward"    # Index finger
    elif fingers == [1,0,0,0,0]: return "right"    # Thumb
    elif fingers == [0,0,0,0,1]: return "left"     # Little finger
    elif fingers == [0,1,1,0,0]: return "reverse"  # Index + Middle
    else: return "stop"

# 📸 Use your phone as webcam by providing PHONE_CAM_URL here
cap = cv2.VideoCapture(PHONE_CAM_URL)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Check your phone IP/connection.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        cmd = "stop"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cmd = detect_gesture(hand_landmarks)

        send_command(cmd)

        cv2.putText(frame, f"Command: {cmd}", (10,40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Rover Control (Phone Cam)", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
