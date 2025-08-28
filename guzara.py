from rich import print
from time import sleep
import sys

def type_text(text, delay=0.1):
    """Type out text character by character with a delay"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        sleep(delay)
    print()

def printlyrics(a):
    """Display lyrics with typing effect"""
    lyrics = [
        "Tu Hai Pehli Aadat Meri",
        "Aur Aakhri Tu Hi Haiiiiiiiiii.......",
        "Ban Betha Mai Tera Aashiq",
        "Tera Ishq Zaruuri Haiiii.....",
        "O Tere Bina Mar Janaaaaaaaa....",
        "Tere Bina Mar Jana",
        "Tu Hi Saanson Ka Sahara Hai...........",
        "Tere Bina Na Guzaara Aee.........",
        "Tuhi Saanson Ka Sahara Hai..................",
        "Tere Bina Na Guzaara Hai"
    ]
    
    print("[bold blue]Guzaara[/bold blue]")
    sleep(1)
    
    for line in lyrics:
        type_text(f"{line}")
        sleep(0.2)
printlyrics(0)