import cv2
import numpy as np
import time

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Allow camera to warm up
time.sleep(2)

# Capture background frames
print("Capturing background... Stay out of the frame!")
background_frames = 30
frames = []

for _ in range(background_frames):
    ret, frame = cap.read()
    if ret:
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frames.append(frame)
    time.sleep(0.1)

# Create median background
background = np.median(frames, axis=0).astype(np.uint8)
print("Background captured!")

# Define door dimensions (center of frame)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
door_width = int(frame_width * 0.15)  # 15% of frame width
door_height = int(frame_height * 0.8)  # 80% of frame height
door_x = (frame_width - door_width) // 2
door_y = (frame_height - door_height) // 2

# Define cloak color (adjust these values based on your cloak)
lower_cloak = np.array([140, 50, 50])  # Lower HSV bound for pink
upper_cloak = np.array([180, 255, 255])  # Upper HSV bound for pink

# Kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for cloak color
    cloak_mask = cv2.inRange(hsv, lower_cloak, upper_cloak)
    
    # Apply morphological operations to clean up mask
    cloak_mask = cv2.morphologyEx(cloak_mask, cv2.MORPH_OPEN, kernel)
    cloak_mask = cv2.morphologyEx(cloak_mask, cv2.MORPH_DILATE, kernel)
    
    # Find contours to filter out small objects
    contours, _ = cv2.findContours(cloak_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Filter small objects
            cv2.drawContours(cloak_mask, [contour], -1, 0, -1)
    
    # Create regions: left, door, right
    left_region = frame[:, :door_x]
    door_region = frame[:, door_x:door_x+door_width]
    right_region = frame[:, door_x+door_width:]
    
    # Apply invisibility effect only in left region
    left_background = background[:, :door_x]
    left_mask = cloak_mask[:, :door_x]
    
    # Apply bilateral filtering for smooth blending
    left_background = cv2.bilateralFilter(left_background, 9, 75, 75)
    
    # Create invisible effect in left region
    invisible_left = np.where(left_mask[..., None], left_background, left_region)
    
    # Combine all regions
    result = np.hstack((invisible_left, door_region, right_region))
    
    # Draw door outline
    cv2.rectangle(result, (door_x, door_y), (door_x+door_width, door_y+door_height), (0, 0, 255), 3)
    cv2.putText(result, "Doraemon Door", (door_x+10, door_y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add instructions
    cv2.putText(result, "Stand on right, pass through door to become invisible", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result, "Press 'q' to quit", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show result
    cv2.imshow("Doraemon Door Invisibility", result)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
