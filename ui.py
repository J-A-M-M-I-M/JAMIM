# ui.py
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer , Qt
import sys
import cv2
import numpy as np
from hand_tracker import get_finger_status, recognize_gesture
import mediapipe as mp
import time

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Recognition")
        self.setGeometry(100, 100, 800, 600)  # 창 크기 키움

        # UI 구성
        self.label = QLabel()
        self.gesture_label = QLabel("Gesture:")
        self.gesture_label.setFont(QFont("Arial", 20))
        self.gesture_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.label, stretch=8)
        layout.addWidget(self.gesture_label, stretch=1)
        self.setLayout(layout)


        # 카메라 및 mediapipe 초기화
        self.cap = cv2.VideoCapture(0)
        print("🎥 VideoCapture opened:", self.cap.isOpened())
        time.sleep(2)  # wait for camera to warm up
        for _ in range(10):
            ret, frame = self.cap.read()
            print(f"ret: {ret}, frame is None: {frame is None}")
        ret, frame = self.cap.read()
        if not self.cap.isOpened():
            print("❌ 카메라를 열 수 없습니다.")
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # 타이머 시작
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30fps 정도

    def update_frame(self):
        frame = self.get_frame()
        if frame is not None:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qimg))

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("❌ 프레임을 읽을 수 없습니다.")
            return None
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                finger_status = get_finger_status(hand_landmarks)
                gesture = recognize_gesture(finger_status)
                if gesture:
                    self.gesture_label.setText(f"gesture: {gesture}")
        return image

    def closeEvent(self, event):
        self.cap.release()  # 앱 종료 시 카메라 해제
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
