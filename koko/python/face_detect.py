import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

with mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
) as face_detection:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Camera not found")
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        # Draw guide lines
        cv2.line(frame, (w//3, 0), (w//3, h), (255, 0, 0), 2)
        cv2.line(frame, (2*w//3, 0), (2*w//3, h), (255, 0, 0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb)

        position = "NO FACE"

        if results.detections:

            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)

                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)

                center_x = x + box_w // 2
                center_y = y + box_h // 2

                # Determine position
                if center_x < w // 3:
                    position = "LEFT"

                elif center_x > (2 * w // 3):
                    position = "RIGHT"

                else:
                    position = "CENTER"

                # Face rectangle
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + box_w, y + box_h),
                    (0, 255, 0),
                    2
                )

                # Face center point
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (0, 0, 255),
                    -1
                )

                # Show coordinates
                cv2.putText(
                    frame,
                    f"X:{center_x}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Display position
        cv2.putText(
            frame,
            f"POSITION: {position}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )

        cv2.imshow("Face Following Robot Vision", frame)

        key = cv2.waitKey(1)

        if key == 27:  # ESC key
            break

cap.release()
cv2.destroyAllWindows()