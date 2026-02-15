import cv2  # python's open cv library
import mediapipe as mp #machine learning library (3.9 aur 3.10 ke niche wale version mein he support krti hain)
import requests
import time

ESP_IP = "10.196.141.115"  #your kocal IP 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_gesture(hand):
    tips = [4,8,12,16,20]  # Thumb, Index, Middle, Ring, Little
    fingers = [0,0,0,0,0]

    # Thumb
    if hand.landmark[tips[0]].x < hand.landmark[tips[0]-1].x:
        fingers[0] = 1
    # Other fingers
    for i in range(1,5):
        if hand.landmark[tips[i]].y < hand.landmark[tips[i]-2].y:
            fingers[i] = 1

    # logic
    if fingers == [0,1,0,0,0]:
        return "forward"
    elif fingers == [0,1,1,0,0]:
        return "reverse"
    elif fingers == [1,0,0,0,0]:
        return "left"
    elif fingers == [0,0,0,0,1]:
        return "right"
    else:
        return "stop"

# Camera access
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
prev_cmd = ""

while True:
    ret, frame = cap.read()
    if not ret: continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    cmd = "stop"
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cmd = detect_gesture(hand_landmarks)

    # Send command only if changed
    if cmd != prev_cmd:
        for attempt in range(3):
            try:
                r = requests.get(f"http://{ESP_IP}/{cmd}", timeout=0.9)
                if r.status_code == 200: print("Sent:", cmd)  
                break
            except requests.exceptions.RequestException as e:
                print(f"ESP32 NOT REACHABLE ({attempt+1}):", e) #error
                time.sleep(0.2)
        prev_cmd = cmd

    cv2.putText(frame, f"Command: {cmd}", (10,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("Gesture Robot", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
