# hand_tracker.py
import cv2
import mediapipe as mp

def get_finger_status(landmarks):
    finger_status = []
    fingers = [
        ("thumb", 4, 2),
        ("index", 8, 6),
        ("middle", 12, 10),
        ("ring", 16, 14),
        ("pinky", 20, 18),
    ]
    for name, tip, pip in fingers:
        if name == "thumb":
            tip_x = landmarks.landmark[tip].x
            pip_x = landmarks.landmark[pip].x
            finger_status.append(tip_x < pip_x)
        else:
            tip_y = landmarks.landmark[tip].y
            pip_y = landmarks.landmark[pip].y
            finger_status.append(tip_y < pip_y)
    return finger_status

def recognize_gesture(finger_status):
    thumb, index, middle, ring, pinky = finger_status
    if index and middle and not ring and not pinky:
        return "가위"
    elif not index and not middle and not ring and not pinky:
        return "바위"
    elif index and middle and ring and pinky:
        return "보"
    else:
        return None
