import tkinter as tk
import tkinter.font as tkFont # 引入字体模块
from tkinter.messagebox import showinfo, showwarning, showerror
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import time
import random


WINWIDTH = 400
WINHEIGHT = 400
WINX = 200
WINY = 50

img_x = 120
img_y = 50

question_y = 100

button_width = 100
button_height = 40
yes_button_x = img_x - button_width // 2
no_button_x = WINWIDTH - img_x - button_width//2
button_y = 200

question = "今天是你生日吗？"
yes = "不是"
no = "是的"
truly_yes = "真的是"
truly_no = "真的不是"
title = "present"

win = tk.Tk()
win.title(title)
win.geometry("%sx%s+%s+%s" % (WINWIDTH, WINHEIGHT, WINX, WINY))

quesft = tkFont.Font(family="微软雅黑", size=16, weight=tkFont.BOLD)
q = tk.Label(win, text=question, font=quesft)
q.place(x=img_x, y=question_y)


# 按钮
def clickyes():
    yes_reply = "哈哈哈，开个玩笑"
    showinfo(title = "joke", message = yes_reply)
    question = "今天真真真的是你生日？"
    q = tk.Label(win, text=question, font=quesft)
    q.place(x=img_x, y=question_y)
    win.update()
    # time.sleep(3)

# 不是(按钮)
yes_button = tk.Button(win, text=yes, command=clickyes)
yes_button.place(x=yes_button_x, y=button_y, width=button_width, height=button_height)
# 是的(按钮)
no_button = tk.Button(win, text=no)
no_button.place(x=no_button_x, y=button_y, width=button_width, height=button_height)
# 真的是(按钮)
t_yes_button = tk.Button(win, text=truly_yes)
# 真的不是
t_no_button = tk.Button(win, text=truly_no)
# 移动按钮(开个玩笑)
def mouse_in_no(event):
    bx, by = random.randint(button_width, WINWIDTH-button_width), random.randint(button_height, WINHEIGHT-button_height)
    no_button.place(x=bx, y=by)

no_button.bind("<Motion>", mouse_in_no)

win.mainloop()
