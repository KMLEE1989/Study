from tkinter import *
from tkinter import ttk

import tkinter

win = tkinter.Tk()
win.geometry('300x150')
label = tkinter.Label(win, text='라벨 입니다.')
label.place(x=5, y=5, width=290, height=30)
text= tkinter.Entry(win)
text.place(x=5, y=35, width=290, height=30)
button = tkinter.Button(win, text='버튼 입니다.')
button.place(x=5, y=70, width=290, height=30)
win.mainloop()