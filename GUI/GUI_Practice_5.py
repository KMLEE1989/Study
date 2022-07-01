from tkinter import *
from tkinter import ttk

import tkinter

win = tkinter.Tk()
win.geometry('300x150')
label = tkinter.Label(win,text='라벨 입니다.')
label.pack()
text= tkinter.Entry(win)
text.pack()
button = tkinter.Button(win, text='버튼 입니다.')
button.pack()
win.mainloop()