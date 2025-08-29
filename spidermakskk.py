import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Face contour indices (MediaPipe Face Mesh)
FACE_CONTOUR = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Eye landmarks indices
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

def draw_spiderman_mask(frame, landmarks):
    h, w, _ = frame.shape
    points = []
    
    # Get face contour points
    for idx in FACE_CONTOUR:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        points.append((x, y))
    
    # Create face mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points)], 255)
    
    # Create red face overlay
    red_face = np.zeros_like(frame)
    red_face[:] = (0, 0, 255)  # Red in BGR
    
    # Apply red face to mask area
    face_area = cv2.bitwise_and(red_face, red_face, mask=mask)
    frame = cv2.addWeighted(frame, 1, face_area, 0.7, 0)
    
    # Draw web pattern
    center_x = int(np.mean([p[0] for p in points]))
    center_y = int(np.mean([p[1] for p in points]))
    
    # Radial lines from center to contour points
    for point in points:
        cv2.line(frame, (center_x, center_y), point, (0, 0, 0), 1)
    
    # Concentric web circles
    for scale in [0.3, 0.5, 0.7]:
        scaled_points = []
        for x, y in points:
            # Scale points towards center
            scaled_x = center_x + (x - center_x) * scale
            scaled_y = center_y + (y - center_y) * scale
            scaled_points.append((int(scaled_x), int(scaled_y)))
        
        # Draw scaled contour
        cv2.polylines(frame, [np.array(scaled_points)], True, (0, 0, 0), 1)
    
    # Draw eye areas
    for eye_indices in [LEFT_EYE, RIGHT_EYE]:
        eye_points = []
        for idx in eye_indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            eye_points.append((x, y))
        
        # Create convex hull for eye
        hull = cv2.convexHull(np.array(eye_points))
        
        # Create eye mask
        eye_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(eye_mask, [hull], 255)
        
        # Create white eye overlay
        white_eye = np.zeros_like(frame)
        white_eye[:] = (255, 255, 255)  # White in BGR
        
        # Apply white eye to eye area
        eye_area = cv2.bitwise_and(white_eye, white_eye, mask=eye_mask)
        frame = cv2.addWeighted(frame, 1, eye_area, 0.9, 0)
        
        # Draw eye outline
        cv2.drawContours(frame, [hull], -1, (0, 0, 0), 1)
    
    return frame

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    # Draw Spider-Man mask if face is detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            frame = draw_spiderman_mask(frame, face_landmarks.landmark)
    
    # Display result
    cv2.imshow('Spider-Man Mask', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()