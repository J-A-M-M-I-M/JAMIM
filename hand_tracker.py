import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# MediaPipe Hands 초기화 (정확도 높고 실시간 사용)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
cap = cv2.VideoCapture(0)  # 웹캠 열기

def get_finger_status(landmarks):
    finger_status = []

    # 관절 인덱스 (Mediapipe 기준)
    # [손가락 이름, tip index, pip index]
    fingers = [
        ("thumb", 4, 2),
        ("index", 8, 6),
        ("middle", 12, 10),
        ("ring", 16, 14),
        ("pinky", 20, 18),
    ]

    for name, tip, pip in fingers:
        tip_y = landmarks.landmark[tip].y
        pip_y = landmarks.landmark[pip].y
        if name == "thumb":
            # 엄지는 x좌표로 판별
            tip_x = landmarks.landmark[tip].x
            pip_x = landmarks.landmark[pip].x
            finger_status.append(tip_x < pip_x)  # True면 펴짐
        else:
            finger_status.append(tip_y < pip_y)  # True면 펴짐

    return finger_status  # [엄지, 검지, 중지, 약지, 새끼]

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



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR → RGB 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    # 다시 RGB → BGR 변환
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 관절 그리기
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            finger_status = get_finger_status(hand_landmarks)
            gesture = recognize_gesture(finger_status)
            if gesture:
                cv2.putText(image, f"Gesture: {gesture}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                print("인식된 손동작:", gesture)

    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
