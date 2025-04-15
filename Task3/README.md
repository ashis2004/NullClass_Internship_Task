# Sign Language Detection System

This application provides a GUI interface for sign language detection with both image upload and real-time video capabilities. The system is operational only during specific hours (6 PM to 10 PM).

## Features

- Real-time sign language detection using webcam
- Image upload capability for static sign language detection
- Time-based access control (6 PM - 10 PM)
- Hand landmark visualization
- User-friendly GUI interface

## Requirements

- Python 3.7 or higher
- PyQt5
- OpenCV
- MediaPipe
- TensorFlow
- NumPy
- Pillow

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python sign_language_detector.py
```

2. The application will start and check if the current time is within the allowed hours (6 PM - 10 PM)
3. Use the following buttons:
   - "Start Camera": Begin real-time sign language detection using your webcam
   - "Upload Image": Upload and process a static image for sign language detection
   - "Stop Camera": Stop the real-time detection

## Notes

- The system is only operational between 6 PM and 10 PM
- Make sure you have a working webcam for real-time detection
- For best results, ensure good lighting and clear hand gestures
- The application uses MediaPipe for hand landmark detection

## License

This project is licensed under the MIT License - see the LICENSE file for details. 