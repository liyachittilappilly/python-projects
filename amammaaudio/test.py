import vlc
import time

audio = r"C:\Users\ADMIN\OneDrive\Desktop\amammaaudio\WhatsApp Ptt 2026-07-06 at 11.09.31 AM.ogg"

player = vlc.MediaPlayer(audio)

print("Playing...")
player.play()

time.sleep(10)