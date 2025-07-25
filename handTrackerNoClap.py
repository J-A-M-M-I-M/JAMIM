import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

# 학습된 모델
model = load_model('gesture_lstm_model.keras')

with open('label2idx.pkl', 'rb') as f:
    label2idx = pickle.load(f)
idx2label = {v: k for k, v in label2idx.items()}
seq_length = 15
sequence = deque(maxlen=seq_length)

# MediaPipe 카메라 준비
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.flip(img, 1)
    result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks:
        for res in result.multi_hand_landmarks:
            joint = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark]).flatten()
            sequence.append(joint)
            if len(sequence) == seq_length:
                input_data = np.expand_dims(np.array(sequence), axis=0)  # (1, 15, 63)
                pred = model.predict(input_data)
                idx = np.argmax(pred)
                gesture_name = idx2label[idx]
                print("실시간 인식 결과:", gesture_name)
    cv2.imshow('Test', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
