import pygame
import time

pygame.init()
pygame.mixer.init()

pygame.mixer.music.load("Lost stars.mp3")
pygame.mixer.music.set_volume(0.5)
pygame.mixer.music.play()
time.sleep(5)