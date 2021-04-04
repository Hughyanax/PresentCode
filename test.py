import tkinter as tk

win = tk.Tk()
win.geometry("80x80+20+20")
label = tk.Label(win, text="你好")
label.grid(row=2, column=2)
button1 = tk.Button(win, text="hello world")
button2 = tk.Button(win, text="hello world!")
button1.grid(row=1, column=2)
button2.grid(row=1, column=3)
win.mainloop()