import random
import tkinter as tk
import tkinter.font as tkFont  # 引入字体模块
from tkinter import ttk
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
from pygame import *
from sys import exit

WIN_WIDTH = 600
WIN_HEIGHT = 450
WIN_X = 200
WIN_Y = 50
img_x = 120
img_y = 50
label_width = 30
question_y = 100
button_width = 100
button_height = 40
yes_button_x = img_x - button_width // 2
no_button_x = WIN_WIDTH - img_x - button_width // 2
button_y = 200

question1 = "今天是你生日吗？"
question2 = "今天真的是你生日？"
question3 = "祝福还是要说的, 那四个字要看吗？"
question4 = "扫一下码！"
question5 = "你觉得这就结束了吗?"
question6 = "先放首歌"
answer1 = "行吧, 再见"
want = "想看"
not_want = "不想看"
yes = "是的"
no = "不是"
truly_yes = "真的是"
truly_no = "真的不是"
it_ended = "是的"
not_yet = "还没有"
finish1 = "扫好了"
finish2 = "我就不看"
end = "不想听了"
title = "present"

win = tk.Tk()
win.title(title)
win.geometry("%sx%s+%s+%s" % (WIN_WIDTH, WIN_HEIGHT, WIN_X, WIN_Y))
question_font = tkFont.Font(family="微软雅黑", size=16, weight=tkFont.BOLD)


def make_label(window=win,
               l_t=None,
               font=question_font,
               bg_color=None,
               l_w=label_width):
    return tk.Label(window, text=l_t, font=font, bg=bg_color, width=l_w)


q1 = make_label(l_t=question1, bg_color="lightblue")
q1.pack()
q2 = make_label(l_t=question2, bg_color="lightgreen")
q3 = make_label(l_t=question3, bg_color="lightyellow")
q4 = make_label(l_t=question4, bg_color="lightblue")
q5 = make_label(l_t=question5, bg_color="lightgreen")
q6 = make_label(l_t=question6, bg_color="lightyellow")
q7 = make_label(bg_color="lightblue")
q8 = make_label(l_t=question5, bg_color="lightgreen")

photo = Image.open("my_qrcode.png")  # 括号里为需要显示在图形化界面里的图片
photo = photo.resize((200, 200))
img0 = ImageTk.PhotoImage(photo)
img1 = ttk.Label(text="照片:", image=img0)
handle = {"control": 1}


# 按钮
def click_no():
    """第一次选择，no"""
    showinfo(title="This is a joke", message="哈哈哈，开个玩笑")
    q2.pack()
    button2.config(text=truly_yes, command=click_t_yes)
    button2.place(x=yes_button_x,
                  y=button_y,
                  width=button_width,
                  height=button_height)
    button1.destroy()
    button3.place(x=no_button_x,
                  y=button_y,
                  width=button_width,
                  height=button_height)
    win.update()


def mouse_in_no(event):
    """开玩笑"""
    button1.place(x=random.randint(button_width, WIN_WIDTH - button_width),
                  y=random.randint(button_height, WIN_HEIGHT - button_height))


def click_t_no():
    """ 直接退出 """
    showinfo(title="bye", message="那行吧, 再见")
    exit()


def click_t_yes():
    """ 第二次选择 """
    button2.config(text=want, command=click_want)
    button3.config(text=not_want, command=click_not_want)
    showinfo(title="good", message="那我要开始了")
    q3.pack()
    win.update()


def click_want():
    """ 表示想看 """
    showinfo(title="force", message="算你识相")
    button2.config(text=finish1, command=click_finish)
    button3.config(text=finish2, command=click_finish)
    q4.pack()
    img1.pack()
    win.update()


def click_not_want():
    """ 表示不想看 """
    showinfo(title="simple", message="不看也得看")
    button2.config(text=finish1, command=click_finish)
    button3.config(text=finish2, command=click_finish)
    q4.pack()
    img1.pack()
    win.update()


def click_finish():
    """ 扫好了 """
    img1.destroy()
    q5.pack()
    button2.config(text=it_ended, command=click_ended)
    button3.config(text=not_yet, command=click_not_yet)


def click_not_yet():
    showinfo(title="you are right", message="猜对了, 后面还有呢！")
    q6.pack()
    button4.pack()
    button2.config(text="暂停", command=pause_music)
    button3.config(text="播放", command=continue_music)
    play_music()


def click_ended():
    showinfo(title="you are wrong", message="错了, 后面还有呢！")
    q6.pack()
    button4.pack()
    button2.config(text="暂停", command=pause_music)
    button3.config(text="播放", command=continue_music)
    play_music()


def play_music(path='Lost stars.mp3'):
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    # pygame.mixer.music.set_volume(0.5)
    pygame.mixer.music.play()


handle["control"] = 0


def pause_music():
    if not handle["control"]:
        pygame.mixer.music.pause()
        handle["control"] = 1
    else:
        showinfo(title="已经停了", message="音乐已经停了！")


def continue_music():
    if handle["control"]:
        pygame.mixer.music.play()
        handle["control"] = 0
    else:
        showinfo(title="playing", message="音乐已经播放了")


def click_t_ended():
    text = "猜对了\n欢乐的时光总是短暂的\n这次真的要说再见啦\n再说一句生日快乐, 拜拜！"
    showinfo(title="you are right", message=text)
    exit()


def click_t_not_yet():
    text = "猜错了\n天下无不散的宴席\n这次真的要结束了\n再次祝你生日快乐, 再见！"
    showinfo(title="you are wrong", message=text)
    exit()


def end_music():
    pygame.mixer.music.pause()
    button4.pack_forget()
    q8.pack()
    button2.config(text=it_ended, command=click_t_ended)
    button3.config(text=not_yet, command=click_t_not_yet)


# 第一次选择(是和不是)
button1 = tk.Button(win, text=yes)
button1.place(x=yes_button_x,
              y=button_y,
              width=button_width,
              height=button_height)
button2 = tk.Button(win, text=no, command=click_no)
button2.place(x=no_button_x,
              y=button_y,
              width=button_width,
              height=button_height)
button3 = tk.Button(win, text=truly_no, command=click_t_no)
button4 = tk.Button(win, text=end, command=end_music)
button1.bind("<Motion>", mouse_in_no)
path = 'Lost stars.mp3'
win.update()
win.mainloop()
