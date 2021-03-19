import tkinter as tk
import tkinter.font as tkFont # 引入字体模块
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import time
import random

IMGPATH = "outputpicture.png"
WINWIDTH = 900
WINHEIGHT = 1600
WINX = 400
WINY = 100

img_x = 180
img_y = 60

question_y = 20

button_width = 100
button_height = 40
yes_button_x = img_x - button_width // 2
no_button_x = WINWIDTH - img_x - button_width//2
button_y = 520

image = Image.open(IMGPATH)
i_w, i_h = image.size
def resize( w_box, h_box, pil_image): #参数是：要适应的窗口宽、高、Image.open后的图片
  w, h = pil_image.size #获取图像的原始大小   
  f1 = 1.0*w_box/w 
  f2 = 1.0*h_box/h    
  factor = min([f1, f2])   
  width = int(w*factor)    
  height = int(h*factor)    
  return pil_image.resize((width, height), Image.ANTIALIAS)

resized_image = resize(WINWIDTH, WINHEIGHT, image)
resized_image.save("outputpicture.png")
question = "我喜欢你，有机会吗？"
yes = "有"
no = "没有"
title = "大妹子"


# 新建无法直接关闭的TK类
class NewTk(tk.Tk):
    def destroy(self):
        # 点击界面右上角的关闭按钮时，会调用本函数，
        # 覆盖掉了父类的关闭方法，使得界面无法关闭
        pass


win = NewTk()
win.title(title)
win.geometry("%sx%s+%s+%s" % (WINWIDTH, WINHEIGHT, WINX, WINY))

photo = tk.PhotoImage(file=IMGPATH)
imgLabel = tk.Label(win, image=photo)
imgLabel.place(x=img_x, y=img_y)

quesft = tkFont.Font(family="微软雅黑", size=16, weight=tkFont.BOLD)
q = tk.Label(win, text=question, font=quesft)
q.place(x=img_x, y=question_y)


# 按钮
def clickyes():
    yes_reply = "(*/ω\*)"
    top_width = 100
    top_height = 50

    top = tk.Toplevel()
    # 设置弹出窗口位置和大小
    top_x = WINX+WINWIDTH//2-top_width//2
    top_y = WINY+WINHEIGHT//2-top_height//2
    top_loc = "{}x{}+{}+{}".format(top_width, top_height, top_x, top_y)
    top.geometry(top_loc)

    # 添加内容
    tk.Label(top, text = yes_reply).pack()
    win.update()
    time.sleep(1)
    # 关闭程序
    exit()


yes_button = tk.Button(win, text=yes, command=clickyes)
yes_button.place(x=yes_button_x, y=button_y, width=button_width, height=button_height)


no_button = tk.Button(win, text=no)
no_button.place(x=no_button_x, y=button_y, width=button_width, height=button_height)


def mouse_in_no(event):
    bx, by = random.randint(button_width, WINWIDTH-button_width), random.randint(button_height, WINHEIGHT-button_height)
    no_button.place(x=bx, y=by)

no_button.bind("<Motion>", mouse_in_no)

win.mainloop()
