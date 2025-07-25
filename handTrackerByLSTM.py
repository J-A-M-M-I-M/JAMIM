import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import sounddevice as sd
import keyboard
import pyautogui
import pickle
import threading
import time

# --- 박수 감지 함수 (스레드로 동작하도록 설계) ---
def detect_clap_stream(threshold=0.04, listen_sec=0.5):
    """마이크 스트림에서 Clap(큰 소리) 감지, 임계값/구간 조절 가능"""
    duration = listen_sec  # (초) 단위로 소리 측정
    fs = 16000  # 샘플링 레이트
    audio = sd.rec(int(duration*fs), samplerate=fs, channels=1, blocking=True)
    amplitude = np.max(np.abs(audio))
    if amplitude > threshold:
        print("clap detected!")
        return True
    return False

# --- 동작 인식 활성화 플래그 ---
recognition_enabled = False

def wait_for_clap():
    global recognition_enabled
    while True:
        if not recognition_enabled:
            print("[ Notice ] Clap for motion capture")
            while not recognition_enabled:  # 대기 중이면 계속 박수 감지
                if detect_clap_stream(threshold=0.04, listen_sec=0.5):
                    recognition_enabled = True
                    print("[ Notice ] Motion capture activated")
                    break
        time.sleep(1)  # 너무 빠른 polling 방지

# --- 박수 대기 스레드 시작 ---
clap_thread = threading.Thread(target=wait_for_clap, daemon=True)
clap_thread.start()

# --- LSTM 및 MediaPipe 세팅 (기존 코드 유지) ---
model = load_model('gesture_lstm_model.keras')
with open('label2idx.pkl', 'rb') as f:
    label2idx = pickle.load(f)
idx2label = {v: k for k, v in label2idx.items()}
seq_length = 20
skip_frames = 5
sequence = deque(maxlen=seq_length) 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.flip(img, 1)
    result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    gesture_name = None

    # 박수 인식 상태: 준비시간, 안내 후 시퀀스 캡처
    if recognition_enabled:

        prepare_time = 0.7   # (초) 준비 대기
        start_time = time.time()
        while time.time() - start_time < prepare_time:
            ret2, prep_img = cap.read()
            if not ret2:
                break
            prep_img = cv2.flip(prep_img, 1)
            cv2.putText(prep_img, "Prepare for motion!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,0,0), 2)
            cv2.imshow('Test', prep_img)
            cv2.waitKey(1)

        temp_sequence = []
        frames_collected = 0
        while frames_collected < seq_length:
            ret3, cap_img = cap.read()
            if not ret3:
                break
            cap_img = cv2.flip(cap_img, 1)
            cap_result = hands.process(cv2.cvtColor(cap_img, cv2.COLOR_BGR2RGB))
            if cap_result.multi_hand_landmarks:
                for res in cap_result.multi_hand_landmarks:
                    joint = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark]).flatten()
                    temp_sequence.append(joint)
                    frames_collected += 1
            cv2.putText(cap_img, f"Capturing... {frames_collected}/{seq_length}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,60,200), 2)
            cv2.imshow('Test', cap_img)
            cv2.waitKey(1)

        # 앞부분 프레임(B) 무시, 나머지로 예측
        if len(temp_sequence) == seq_length:
            input_data = np.expand_dims(np.array(temp_sequence[skip_frames:]), axis=0)  # [skip:] 프레임만 인식에 사용 (shape: (1, seq_length-skip, 63))
            # 입력 길이 변경에 맞게 LSTM 모델 정의 필요!
            pred = model.predict(input_data, verbose=0)
            idx = np.argmax(pred)
            gesture_name = idx2label[idx]
            print(f"Captured gesture: {gesture_name}")

            if gesture_name == 'left_swipe':
                keyboard.press_and_release('ctrl+windows+right')
            elif gesture_name == 'make_fist':
                keyboard.press_and_release('space')
            elif gesture_name == 'erm':
                keyboard.press_and_release('f12')   

            # 여기서 단축키 실행 등 원하는 동작 연결
        recognition_enabled = False  # 인식 완료 후 다시 박수 대기

    else:
        # 박수 대기중 안내
        cv2.putText(img, "Clap to activate motion capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 2)
        cv2.imshow('Test', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
