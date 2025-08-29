import cv2
import mediapipe as mp
h = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
c = cv2.VideoCapture(0)
while True:
    r, f = c.read()
    if not r: break
    f = cv2.flip(f, 1)
    res = h.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        for l in res.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(f, l, mp.solutions.hands.HAND_CONNECTIONS)
    cv2.imshow("Hand Tracking", f)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
c.release()
cv2.destroyAllWindows()