import operator
from random import randint

import tensorflow
from flask import Flask

from data import config
from data.conversion import convert_pattern
from data.pattern import add_pattern, load_all
from engine.network import Network
from engine.runner import get_output
from gui.board import Board
from rest.server import NeuralRestEnginge


def save_signature(name, n=-1):
    add_pattern(name, Board().capture(n))


def test_network():
    with tensorflow.Session() as sess:
        all_data = load_all()
        num_sets = len(all_data)
        neural = Network(sess, config.SIZE * 2, num_sets)
        for current_set, pattern_set in enumerate(all_data):
            for pattern in pattern_set[1]:
                if randint(0, 3) == 0:
                    neural.add_test(list(convert_pattern(pattern)), get_output(current_set, num_sets))
                else:
                    neural.add_train(list(convert_pattern(pattern)), get_output(current_set, num_sets))

        neural.train(100000)
        print(neural.log())

        def cb(sign):
            answer = neural.use(sign)[0]
            max_index, max_value = max(enumerate(answer), key=operator.itemgetter(1))
            print(all_data[max_index][0], max_value)

        Board(cb).capture(-1)


if __name__ == '__main__':
    test_network()
