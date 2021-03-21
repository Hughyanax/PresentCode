import tkinter as tk
import tkinter.font as tkFont
from tkinter.messagebox import showinfo
import random

class Window(object):
    def __init__(self, win, text_title, width=400, height=400, x=200, y=50):
        self.win = win
        win.title(text_title)
        win.geometry("%sx%s+%s+%s" % (width, height, x, y))
        frame = tk.Frame(win)
        frame.pack

    def label_place(self, text1, x_pot, y_pot):
        """给窗口增加文本标签, 字体为微软雅黑"""
        label_font = tkFont.Font(family="微软雅黑", size=16, weight=tkFont.BOLD)
        lf = tk.Label(self.win, text=text1, font=label_font)
        lf.place(x=x_pot, y=y_pot)

    def button_place(self, text2, x_pot, y_pot, function, button_width=100, button_height=40):
        """
        给窗口添加按钮，传入参数为：
        按钮文本, x坐标, y坐标, 按钮宽度(默认100), 按钮高度(默认40)
        """
        button = tk.Button(self.win, text=text2, command=function)
        button.place(x=x_pot, y=y_pot, width=button_width, height=button_height)


def clickyes():
    yes_reply = "哈哈哈，开个玩笑"
    showinfo(title = "This is a joke", message = yes_reply)
    win.update()
win = tk.Tk()
app = Window(win, "this is a present")
app.label_place("hello", 20, 20)
app.button_place("是的", 40, 40, clickyes)

win.mainloop()