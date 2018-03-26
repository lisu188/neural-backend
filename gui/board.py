import json
import time
import tkinter as tk
from functools import partial

import requests


class Board:
    def __init__(self, callback=lambda x: None):
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=2000, height=1000)
        self.canvas.pack()
        self.all_coords = []
        self.in_progress = False
        self.coords = []
        self.callback = callback

    def motion(self, n, event):
        if event.state == 256:
            if not self.in_progress:
                self.in_progress = True
                self.start_time = time.time()
            self.coords.append((event.x, event.y, time.time() - self.start_time))
            x1, y1 = (event.x - 1), (event.y - 1)
            x2, y2 = (event.x + 1), (event.y + 1)
            self.canvas.create_oval(x1, y1, x2, y2)
        elif event.state == 0 and self.in_progress and len(self.coords):
            self.in_progress = False
            self.all_coords.append(self.coords)
            self.callback(self.coords)
            self.coords = []
            self.canvas.delete("all")
        if len(self.all_coords) == n:
            self.root.quit()

    def capture(self, n=-1):
        self.canvas.bind('<Motion>', partial(self.motion, n))
        self.root.mainloop()
        return self.all_coords


if __name__ == '__main__':
    def cb(data):
        print(requests.post("http://127.0.0.1:5000/use", data=json.dumps(data)).json())


    Board(cb).capture(-1)
