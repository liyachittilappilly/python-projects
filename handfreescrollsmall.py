import cv2
import mediapipe as mp
import pyautogui
import time

class FaceScrollController:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
        self.cap = cv2.VideoCapture(0)
        self.is_running = False
        self.prev_y = None
        self.threshold = 15
        self.scroll_amount = 85
        self.last_scroll = 0
        self.cooldown = 0.1

    def run(self):
        print("Face Scroll Controller - SPACE to toggle, ESC to exit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            # Status display
            status = "ACTIVE" if self.is_running else "INACTIVE"
            color = (0, 255, 0) if self.is_running else (0, 0, 255)
            cv2.putText(frame, f"Scrolling: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if results.multi_face_landmarks:
                # Get nose tip position (landmark 1)
                nose = results.multi_face_landmarks[0].landmark[1]
                y = int(nose.y * frame.shape[0])
                x = int(nose.x * frame.shape[1])
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
                
                if self.is_running and self.prev_y is not None:
                    movement = self.prev_y - y
                    current_time = time.time()
                    
                    if current_time - self.last_scroll > self.cooldown:
                        if movement > self.threshold:
                            pyautogui.scroll(self.scroll_amount)
                            self.last_scroll = current_time
                            cv2.putText(frame, "SCROLL UP", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        elif movement < -self.threshold:
                            pyautogui.scroll(-self.scroll_amount)
                            self.last_scroll = current_time
                            cv2.putText(frame, "SCROLL DOWN", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                self.prev_y = y
            
            cv2.imshow("Face Scroll Controller", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE
                self.is_running = not self.is_running
                self.prev_y = None
                print(f"Scrolling {'ACTIVATED' if self.is_running else 'DEACTIVATED'}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed")

if __name__ == "__main__":
    try:
        app = FaceScrollController()
        app.run()
    except Exception as e:
        print(f"Error: {e}")