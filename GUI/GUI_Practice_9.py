from tkinter import *
from tkinter import ttk

import tkinter
from tkinter import messagebox


class clock(tkinter.Tk):
    def __init__(self):
        tkinter.Tk.__init__(self)
        self.x = 150
        self.y = 150
        self.geometry('300x300')
        self.length = 30
        self.canvas = tkinter.Canvas(self, bg='white')
        self.canvas.pack(expand='yes', fill='both')
        
if __name__ == '__main__':
    clock = clock()
    while True:
        clock.update()