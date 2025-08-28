import cv2
import mediapipe as mp
import math
import numpy as np
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Web effect particles
class WebParticle:
    def __init__(self, x, y, angle, speed):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.life = 100
        self.size = random.randint(2, 5)
        
    def update(self):
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        self.life -= 2
        self.speed *= 0.98  # Slow down over time
        
    def draw(self, img):
        if self.life > 0:
            alpha = min(255, self.life * 2)
            cv2.circle(img, (int(self.x), int(self.y)), self.size, (255, 255, 255), -1)

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Function to check if fingers are extended
def is_finger_extended(landmarks, tip_idx, pip_idx, wrist_idx, orientation_up):
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    wrist = landmarks[wrist_idx]
    
    if orientation_up:
        return tip.y < pip.y
    else:
        return tip.y > pip.y

# Main function
def main():
    cap = cv2.VideoCapture(0)
    web_particles = []
    web_active = False
    last_web_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        # Update and draw web particles
        for particle in web_particles[:]:
            particle.update()
            if particle.life <= 0:
                web_particles.remove(particle)
            else:
                particle.draw(frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = hand_landmarks.landmark
                
                # Get hand orientation (upright or inverted)
                wrist = landmarks[0]
                middle_base = landmarks[9]
                orientation_up = middle_base.y > wrist.y
                
                # Check Spiderman pose:
                # 1. Thumb and index finger touching
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                thumb_index_dist = calculate_distance(thumb_tip, index_tip)
                
                # 2. Middle, ring, and pinky fingers extended
                middle_extended = is_finger_extended(landmarks, 12, 10, 0, orientation_up)
                ring_extended = is_finger_extended(landmarks, 16, 14, 0, orientation_up)
                pinky_extended = is_finger_extended(landmarks, 20, 18, 0, orientation_up)
                
                # Activate web if pose detected
                if (thumb_index_dist < 0.05 and 
                    middle_extended and ring_extended and pinky_extended):
                    
                    # Get web origin (wrist position)
                    h, w, _ = frame.shape
                    web_origin = (int(landmarks[0].x * w), int(landmarks[0].y * h))
                    
                    # Get direction (from wrist to middle finger tip)
                    middle_tip = landmarks[12]
                    direction_x = middle_tip.x - wrist.x
                    direction_y = middle_tip.y - wrist.y
                    direction_angle = math.atan2(direction_y, direction_x)
                    
                    # Create new web particles
                    current_time = cv2.getTickCount() / cv2.getTickFrequency()
                    if current_time - last_web_time > 0.05:  # Limit particle creation rate
                        for _ in range(5):
                            # Add some randomness to the direction
                            angle_offset = random.uniform(-0.3, 0.3)
                            particle_angle = direction_angle + angle_offset
                            particle_speed = random.uniform(10, 20)
                            
                            web_particles.append(
                                WebParticle(
                                    web_origin[0], 
                                    web_origin[1],
                                    particle_angle,
                                    particle_speed
                                )
                            )
                        last_web_time = current_time
                    
                    web_active = True
                else:
                    web_active = False
        
        # Draw web shooting effect if active
        if web_active:
            # Draw a bright white circle at the wrist
            h, w, _ = frame.shape
            wrist_pos = (int(landmarks[0].x * w), int(landmarks[0].y * h))
            cv2.circle(frame, wrist_pos, 10, (255, 255, 255), -1)
            
            # Add a glow effect
            for radius in range(15, 30, 5):
                alpha = max(0, 100 - radius * 3)
                cv2.circle(frame, wrist_pos, radius, (255, 255, 255), 1)
        
        # Display instructions
        cv2.putText(frame, "Make Spiderman web-shooting pose!", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Spiderman Web Shooter', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()