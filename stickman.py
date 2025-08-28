import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, 
                    model_complexity=1)  # Enable full body model for more landmarks
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set up the figure dimensions
FIGURE_WIDTH = 300
FIGURE_HEIGHT = 500

# Load the face image (replace 'face.png' with your image path)
face_img = cv2.imread('https://i.pinimg.com/736x/70/1f/e8/701fe8ac0ec65f3d20e20ff89f78767c.jpg', cv2.IMREAD_UNCHANGED)
if face_img is None:
    print("Error: Could not load face image. Using default circle.")
    use_face_img = False
else:
    # Resize face image to appropriate size
    face_img = cv2.resize(face_img, (30, 30))
    use_face_img = True

def draw_stick_figure(image, landmarks):
    """Draw a stick figure that mirrors the user's pose with all landmarks marked"""
    # Create a blank canvas for the stick figure
    stick_figure = np.zeros((FIGURE_HEIGHT, FIGURE_WIDTH, 3), dtype=np.uint8)
    stick_figure.fill(240)  # Light gray background
    
    if not landmarks:
        return stick_figure
    
    # Convert normalized landmarks to pixel coordinates in the stick figure space
    def to_stick_coords(landmark):
        x = int(landmark.x * FIGURE_WIDTH)
        y = int(landmark.y * FIGURE_HEIGHT)
        return (x, y)
    
    # Define connections between landmarks (full body)
    connections = [
        # Face connections
        (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE),
        (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.NOSE),
        (mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.NOSE),
        (mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.LEFT_EYE),
        (mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.RIGHT_EYE),
        
        # Head to shoulders
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER),
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        
        # Torso
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
        
        # Left arm
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_PINKY),
        (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX),
        (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_THUMB),
        
        # Right arm
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_PINKY),
        (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX),
        (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_THUMB),
        
        # Left leg
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL),
        (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
        
        # Right leg
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL),
        (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
    ]
    
    # Draw connections with rose color (thicker lines)
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = to_stick_coords(landmarks[start_idx])
            end_point = to_stick_coords(landmarks[end_idx])
            
            # Draw the line with rose color (RGB: 255, 182, 193) and increased thickness
            cv2.line(stick_figure, start_point, end_point, (255, 182, 193), 8)
    
    # Draw all landmarks as small black circles
    for idx, landmark in enumerate(landmarks):
        point = to_stick_coords(landmark)
        cv2.circle(stick_figure, point, 3, (0, 0, 0), -1)  # Small black circles
    
    # Draw head with custom face image
    if mp_pose.PoseLandmark.NOSE < len(landmarks):
        nose_point = to_stick_coords(landmarks[mp_pose.PoseLandmark.NOSE])
        
        if use_face_img:
            # Calculate position to overlay face image
            x, y = nose_point
            h, w = face_img.shape[:2]
            
            # Ensure the face image fits within the stick figure boundaries
            x1 = max(0, x - w//2)
            y1 = max(0, y - h//2)
            x2 = min(FIGURE_WIDTH, x + w//2)
            y2 = min(FIGURE_HEIGHT, y + h//2)
            
            # Adjust the face image if it goes out of bounds
            if x2 - x1 > 0 and y2 - y1 > 0:
                # Calculate the region of interest
                roi = stick_figure[y1:y2, x1:x2]
                
                # Resize face image to fit the ROI
                face_resized = cv2.resize(face_img, (x2 - x1, y2 - y1))
                
                # If face image has alpha channel, use it for transparency
                if face_resized.shape[2] == 4:
                    # Split the color and alpha channels
                    b, g, r, a = cv2.split(face_resized)
                    # Create a mask from the alpha channel
                    mask = a / 255.0
                    # Blend the face image with the background
                    for c in range(3):
                        roi[:, :, c] = roi[:, :, c] * (1 - mask) + face_resized[:, :, c] * mask
                else:
                    # If no alpha channel, simply overlay the image
                    stick_figure[y1:y2, x1:x2] = face_resized
        else:
            # Fallback to yellow circle if face image not available
            cv2.circle(stick_figure, nose_point, 15, (255, 255, 0), -1)
    
    return stick_figure

def calculate_angle(a, b, c):
    """Calculate angle ABC (in degrees)"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def detect_movement(landmarks, prev_landmarks):
    """Detect specific body movements"""
    if not landmarks or not prev_landmarks:
        return "No movement detected"
    
    # Get key landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    
    # Calculate angles
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    # Detect movements
    if left_arm_angle < 70 and right_arm_angle < 70:
        return "Arms raised"
    elif left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
        return "Hands up"
    elif abs(left_hip.y - right_hip.y) > 0.05:
        return "Hip tilt"
    elif left_wrist.y > left_hip.y and right_wrist.y > right_hip.y:
        return "Arms down"
    else:
        return "Neutral position"

# Main loop
prev_landmarks = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect body landmarks
    results = pose.process(rgb_frame)
    
    # Create a copy of the frame for drawing
    output_frame = frame.copy()
    
    # Draw the pose annotation on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            output_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        # Draw the stick figure
        stick_figure = draw_stick_figure(output_frame, results.pose_landmarks.landmark)
        
        # Detect movement
        movement = detect_movement(results.pose_landmarks.landmark, prev_landmarks)
        
        # Display movement text
        cv2.putText(output_frame, f"Movement: {movement}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Store current landmarks for next frame
        prev_landmarks = results.pose_landmarks.landmark
    else:
        # Draw empty stick figure if no landmarks detected
        stick_figure = np.zeros((FIGURE_HEIGHT, FIGURE_WIDTH, 3), dtype=np.uint8)
        stick_figure.fill(240)
        cv2.putText(stick_figure, "No body detected", (50, FIGURE_HEIGHT//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Resize stick figure to match frame height
    stick_figure_resized = cv2.resize(stick_figure, (FIGURE_WIDTH, frame.shape[0]))
    
    # Combine the original frame and the stick figure
    combined_frame = np.hstack((output_frame, stick_figure_resized))
    
    # Display the combined frame
    cv2.imshow('Body Movement Detection', combined_frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()