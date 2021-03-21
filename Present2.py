import tkinter as tk
import tkinter.font as tkFont
from tkinter.messagebox import showinfo
import random

class Window(object):
    def __init__(self, win, text_title):
        win.title(text_title)
        frame = tk.Frame(win)
        frame.pack
win = tk.Tk()
app = Window(win, "this is a present")
win.mainloop()