import sys
import cv2
import numpy as np
from datetime import datetime, time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import mediapipe as mp
import tensorflow as tf

class SignLanguageDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Detector")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize variables
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_running = False
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create display area
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(800, 600)
        self.display_label.setStyleSheet("border: 2px solid black;")
        layout.addWidget(self.display_label)
        
        # Create button layout
        button_layout = QHBoxLayout()
        
        # Create buttons
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)
        
        # Add buttons to layout
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)
        
        # Check if current time is within allowed hours
        self.check_time()
        
    def check_time(self):
        current_time = datetime.now().time()
        allowed_start = time(18, 0)  # 6 PM
        allowed_end = time(22, 0)    # 10 PM
        
        if allowed_start <= current_time <= allowed_end:
            self.start_button.setEnabled(True)
            self.upload_button.setEnabled(True)
            self.status_label.setText("Status: System Active (Within allowed hours)")
        else:
            self.start_button.setEnabled(False)
            self.upload_button.setEnabled(False)
            self.status_label.setText("Status: System Inactive (Outside allowed hours)")
    
    def start_camera(self):
        if not self.check_time():
            return
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open camera!")
            return
            
        self.is_running = True
        self.timer.start(30)  # 30ms = 33fps
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.upload_button.setEnabled(False)
        
    def stop_camera(self):
        self.is_running = False
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.upload_button.setEnabled(True)
        
    def upload_image(self):
        if not self.check_time():
            return
            
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        
        if file_name:
            image = cv2.imread(file_name)
            if image is not None:
                self.process_image(image)
            else:
                QMessageBox.critical(self, "Error", "Could not load image!")
                
    def process_image(self, image):
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
        # Convert to Qt format and display
        self.display_image(image)
        
    def update_frame(self):
        if not self.is_running:
            return
            
        ret, frame = self.cap.read()
        if ret:
            self.process_image(frame)
            
    def display_image(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            image.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.display_label.setPixmap(scaled_pixmap)
        
    def check_time(self):
        current_time = datetime.now().time()
        allowed_start = time(18, 0)  # 6 PM
        allowed_end = time(22, 0)    # 10 PM
        
        if allowed_start <= current_time <= allowed_end:
            return True
        else:
            QMessageBox.warning(
                self,
                "Time Restriction",
                "The system is only available between 6 PM and 10 PM."
            )
            return False
            
    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageDetector()
    window.show()
    sys.exit(app.exec_()) 