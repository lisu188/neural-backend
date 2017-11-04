import sys

from data.pattern import save_pattern, add_pattern
from gui.board import Board
from rest.server import NeuralRestEnginge

if __name__ == '__main__':
    def cb(data):
        add_pattern(sys.argv[1], [data])


    Board(cb).capture(-1)
