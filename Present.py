import tkinter as tk
import tkinter.font as tkFont # 引入字体模块
from tkinter.messagebox import showinfo
import random


WINWIDTH = 600
WINHEIGHT = 450
WINX = 200
WINY = 50

img_x = 120
img_y = 50
label_width = 30

question_y = 100

button_width = 100
button_height = 40
yes_button_x = img_x - button_width // 2
no_button_x = WINWIDTH - img_x - button_width//2
button_y = 200

question1 = "今天是你生日吗？"
question2 = "今天真的是你生日？"
question3 = "祝福还是要说的, 那四个字要看吗？"
question4 = "扫一下码！"
answer1 = "行吧, 再见"
want = "想看"
not_want= "不想看"
yes = "不是"
no = "是的"
truly_yes = "真的是"
truly_no = "真的不是"
title = "present"

win = tk.Tk()
win.title(title)
win.geometry("%sx%s+%s+%s" % (WINWIDTH, WINHEIGHT, WINX, WINY))

quesft = tkFont.Font(family="微软雅黑", size=16, weight=tkFont.BOLD)
q1 = tk.Label(win, text=question1, font=quesft, width=label_width, bg="lightblue")
q1.pack()

handle = {}
handle["control"] = 0
# 按钮
def clickyes():
    yes_reply = "哈哈哈，开个玩笑"
    showinfo(title = "This is a joke", message = yes_reply)
    q2 = tk.Label(win, text=question2, font=quesft, width=label_width, bg="lightgreen")
    q2.pack()
    yes_button.destroy()
    no_button.destroy()
    t_y_button.place(x=yes_button_x, y=button_y, width=button_width, height=button_height)
    t_n_button.place(x=no_button_x, y=button_y, width=button_width, height=button_height)
    win.update()
def mouse_in_no(event):
    bx, by = random.randint(button_width, WINWIDTH-button_width), random.randint(button_height, WINHEIGHT-button_height)
    no_button.place(x=bx, y=by)
def click_t_no():
    bye = "那行吧, 再见"
    showinfo(title="bye", message=bye)
    quit()
def click_t_yes():
    ans_t_yes = "那我要开始了"
    t_n_button.destroy()
    t_y_button.destroy()
    i_want.place(x=yes_button_x, y=button_y, width=button_width, height=button_height)
    i_not_want.place(x=no_button_x, y=button_y, width=button_width, height=button_height)
    showinfo(title="good", message=ans_t_yes)
    q3 = tk.Label(win, text=question3, font=quesft, width=label_width, bg="lightyellow")
    q3.pack()
    win.update()
def click_want():
    ans_want = "算你识相"
    showinfo(title="simple", message=ans_want)
    q4 = tk.Label(win, text=question4, font=quesft, width=label_width, bg="lightyellow")
    q4.pack()
    win.update()
def click_not_want():
    ans_not_want = "不看也得看"
    showinfo(title="force", message=ans_not_want)
    q4 = tk.Label(win, text=question4, font=quesft, width=label_width, bg="lightyellow")
    q4.pack()
    win.update()
# 第一次选择(是和不是)
yes_button = tk.Button(win, text=yes, command=clickyes)
yes_button.place(x=yes_button_x, y=button_y, width=button_width, height=button_height)
no_button = tk.Button(win, text=no)
no_button.place(x=no_button_x, y=button_y, width=button_width, height=button_height)
# 第二次选择
t_y_button = tk.Button(win, text=truly_yes, command=click_t_yes)
t_n_button = tk.Button(win, text=truly_no, command=click_t_no)
# 第三次选择
i_want= tk.Button(win, text=want, command=click_want)
i_not_want= tk.Button(win, text=not_want, command=click_not_want)

no_button.bind("<Motion>", mouse_in_no)
win.update()
win.mainloop()
