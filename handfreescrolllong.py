import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import threading

class FaceScrollController:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Control variables
        self.is_running = False
        self.face_detected = False
        self.prev_face_y = None
        self.scroll_threshold = 15  # Minimum pixels to trigger scroll
        self.scroll_amount = 85      # Scroll amount per trigger
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.1  # Seconds between scrolls
        
        # Create control window
        self.create_control_window()
        
    def create_control_window(self):
        """Create a simple control window with instructions"""
        cv2.namedWindow("Face Scroll Controller", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Scroll Controller", 800, 600)
        
        # Create a blank frame for instructions
        instruction_frame = np.zeros((200, 800, 3), dtype=np.uint8)
        cv2.putText(instruction_frame, "Face Scroll Controller - Real-time Scrolling", 
                   (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(instruction_frame, "Press SPACE to Start/Stop | ESC to Exit", 
                   (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(instruction_frame, "Move your face UP to scroll UP, DOWN to scroll DOWN", 
                   (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(instruction_frame, "Keep your face centered in the frame for best results", 
                   (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow("Face Scroll Controller", instruction_frame)
        
    def process_frame(self, frame):
        """Process each video frame for face detection and scrolling"""
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        # Create a copy for drawing
        output_frame = frame.copy()
        
        # Draw status background
        cv2.rectangle(output_frame, (0, 0), (640, 60), (0, 0, 0), -1)
        
        # Draw status text
        status_text = "SCROLLING: ACTIVE" if self.is_running else "SCROLLING: INACTIVE"
        status_color = (0, 255, 0) if self.is_running else (0, 0, 255)
        cv2.putText(output_frame, status_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        face_status = "FACE DETECTED" if self.face_detected else "NO FACE DETECTED"
        face_color = (0, 255, 0) if self.face_detected else (0, 0, 255)
        cv2.putText(output_frame, face_status, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
        
        if results.multi_face_landmarks:
            self.face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Get nose tip position (landmark 1)
                nose_tip = face_landmarks.landmark[1]
                face_y = int(nose_tip.y * output_frame.shape[0])
                face_x = int(nose_tip.x * output_frame.shape[1])
                
                # Draw tracking point
                cv2.circle(output_frame, (face_x, face_y), 8, (0, 255, 0), -1)
                cv2.circle(output_frame, (face_x, face_y), 12, (0, 255, 0), 2)
                
                # Draw vertical reference line
                cv2.line(output_frame, (face_x, 0), (face_x, output_frame.shape[0]), (0, 255, 0), 1)
                
                # Handle scrolling if enabled
                if self.is_running and self.prev_face_y is not None:
                    self.handle_scroll(face_y, output_frame)
                
                self.prev_face_y = face_y
        else:
            self.face_detected = False
            self.prev_face_y = None
        
        return output_frame
    
    def handle_scroll(self, current_y, frame):
        """Handle scrolling based on face movement"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scroll_time < self.scroll_cooldown:
            return
        
        # Calculate movement (negative means face moved up)
        movement = self.prev_face_y - current_y
        
        # Scroll up if face moves up significantly
        if movement > self.scroll_threshold:
            pyautogui.scroll(self.scroll_amount)
            self.last_scroll_time = current_time
            self.show_scroll_indicator(frame, "UP", (0, 255, 0))
        
        # Scroll down if face moves down significantly
        elif movement < -self.scroll_threshold:
            pyautogui.scroll(-self.scroll_amount)
            self.last_scroll_time = current_time
            self.show_scroll_indicator(frame, "DOWN", (0, 0, 255))
    
    def show_scroll_indicator(self, frame, direction, color):
        """Show scroll direction indicator on frame"""
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
        
        # Blend with original frame
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw direction text
        cv2.putText(frame, f"SCROLL {direction}", (200, frame.shape[0] // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
        cv2.putText(frame, f"SCROLL {direction}", (200, frame.shape[0] // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    
    def run(self):
        """Main application loop"""
        print("Face Scroll Controller Started")
        print("Controls:")
        print("  SPACE - Start/Stop scrolling")
        print("  ESC   - Exit application")
        print("\nPosition your face in the center of the frame")
        print("Move your face up/down to control scrolling")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Show frame
            cv2.imshow("Face Scroll Controller", processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord(' '):  # SPACE key
                self.is_running = not self.is_running
                if self.is_running:
                    print("Scrolling ACTIVATED")
                    # Reset tracking when starting
                    self.prev_face_y = None
                    self.last_scroll_time = 0
                else:
                    print("Scrolling DEACTIVATED")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed")

def main():
    """Main function to run the application"""
    try:
        app = FaceScrollController()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your camera is connected and working")

if __name__ == "__main__":
    main()