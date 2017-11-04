import json

import requests

from gui.board import Board

if __name__ == '__main__':
    def cb(data):
        print(requests.post("http://127.0.0.1:5000/use", data=json.dumps(data)).json())


    Board(cb).capture(-1)
