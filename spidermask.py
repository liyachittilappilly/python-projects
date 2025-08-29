import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

def draw_spiderman_mask(frame, landmarks):
    h, w, _ = frame.shape
    
    # Get key facial points
    forehead = landmarks[10]
    chin = landmarks[152]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    
    # Convert to image coordinates
    cx = (forehead.x + chin.x) / 2 * w
    cy = (forehead.y + chin.y) / 2 * h
    face_width = (right_cheek.x - left_cheek.x) * w
    face_height = (chin.y - forehead.y) * h
    
    # Draw red face mask
    cv2.ellipse(frame, (int(cx), int(cy)), 
                (int(face_width/2), int(face_height/2)), 
                0, 0, 360, (0, 0, 255), -1)
    
    # Draw eye areas
    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
    
    # Process left eye
    left_eye_points = [(landmarks[i].x * w, landmarks[i].y * h) for i in left_eye_indices]
    left_x_min = min(p[0] for p in left_eye_points)
    left_x_max = max(p[0] for p in left_eye_points)
    left_y_min = min(p[1] for p in left_eye_points)
    left_y_max = max(p[1] for p in left_eye_points)
    left_eye_center = ((left_x_min + left_x_max)/2, (left_y_min + left_y_max)/2)
    left_eye_width = (left_x_max - left_x_min) * 1.2
    left_eye_height = (left_y_max - left_y_min) * 1.2
    
    # Draw white left eye
    cv2.ellipse(frame, (int(left_eye_center[0]), int(left_eye_center[1])), 
               (int(left_eye_width/2), int(left_eye_height/2)), 
               0, 0, 360, (255, 255, 255), -1)
    
    # Process right eye
    right_eye_points = [(landmarks[i].x * w, landmarks[i].y * h) for i in right_eye_indices]
    right_x_min = min(p[0] for p in right_eye_points)
    right_x_max = max(p[0] for p in right_eye_points)
    right_y_min = min(p[1] for p in right_eye_points)
    right_y_max = max(p[1] for p in right_eye_points)
    right_eye_center = ((right_x_min + right_x_max)/2, (right_y_min + right_y_max)/2)
    right_eye_width = (right_x_max - right_x_min) * 1.2
    right_eye_height = (right_y_max - right_y_min) * 1.2
    
    # Draw white right eye
    cv2.ellipse(frame, (int(right_eye_center[0]), int(right_eye_center[1])), 
               (int(right_eye_width/2), int(right_eye_height/2)), 
               0, 0, 360, (255, 255, 255), -1)
    
    # Draw web pattern
    # Radial lines
    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        x = cx + face_width/2 * math.cos(rad)
        y = cy + face_height/2 * math.sin(rad)
        cv2.line(frame, (int(cx), int(cy)), (int(x), int(y)), (0, 0, 0), 2)
    
    # Concentric ellipses
    for scale in [0.3, 0.6, 0.9]:
        w_scale = int(face_width * scale / 2)
        h_scale = int(face_height * scale / 2)
        cv2.ellipse(frame, (int(cx), int(cy)), (w_scale, h_scale), 0, 0, 360, (0, 0, 0), 2)

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
            draw_spiderman_mask(frame, face_landmarks.landmark)
    
    # Display result
    cv2.imshow('Spider-Man Mask', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()