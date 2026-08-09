import cv2
import mediapipe as mp
import vlc
import time
import os

# -----------------------
# AUDIO
# -----------------------
audio_file = r"C:\Users\ADMIN\OneDrive\Desktop\amammaaudio\WhatsApp Audio 2026-07-06 at 11.30.49 AM.aac"

if not os.path.exists(audio_file):
    print("Audio file not found!")
    exit()

# -----------------------
# HAND DETECTOR
# -----------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

played = False
player = None


def fingers_up(hand_landmarks):
    lm = hand_landmarks.landmark

    thumb = lm[4].x < lm[3].x
    index = lm[8].y < lm[6].y
    middle = lm[12].y < lm[10].y
    ring = lm[16].y < lm[14].y
    pinky = lm[20].y < lm[18].y

    return thumb and index and middle and ring and pinky


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    hand_open = False

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            if fingers_up(hand_landmarks):
                hand_open = True

    if hand_open:

        cv2.putText(frame,
                    "HIGH FIVE!",
                    (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        if not played:

            print("Playing audio...")

            if player is not None:
                player.stop()
                player.release()

            player = vlc.MediaPlayer(audio_file)
            player.play()

            played = True

    else:
        played = False

    cv2.imshow("High Five Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

if player is not None:
    player.stop()
    player.release()

cv2.destroyAllWindows()