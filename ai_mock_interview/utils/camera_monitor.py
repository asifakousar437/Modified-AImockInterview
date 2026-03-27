import cv2
import base64
import numpy as np

def detect_faces(base64_image):
    try:
        img_data = base64.b64decode(base64_image.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load face cascade with improved parameters
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Use single, more reliable face detection method to avoid false positives
        faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(60, 60))
        
        # Apply additional filtering to reduce false positives
        valid_faces = []
        for (x, y, w, h) in faces:
            aspect_ratio = w / h
            # Strict filtering: reasonable aspect ratio and minimum size
            if 0.8 <= aspect_ratio <= 1.3 and w >= 60 and h >= 60:
                valid_faces.append((x, y, w, h))
        
        face_count = len(valid_faces)
        
        # Debug logging
        print(f"DEBUG: Face detection - Raw faces: {len(faces)}, Valid faces: {face_count}")
        
        return face_count

    except Exception as e:
        print(f"DEBUG: Face detection error: {e}")
        return 0