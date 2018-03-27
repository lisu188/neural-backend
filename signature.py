import operator
from random import randint

import numpy
import tensorflow
import argparse
from data import config
from data.conversion import convert_pattern
from data.pattern import add_pattern, load_all
from engine.network import Network
from engine.runner import get_output
from gui.board import Board


def save_signature(name, n=-1):
    add_pattern(name, Board().capture(n))


def test_network(ite):
    with tensorflow.Session() as sess:
        all_data = load_all()
        num_sets = len(all_data)
        neural = Network(sess, config.SIZE, num_sets)
        neural.add_test([0] * config.SIZE, [0] * num_sets)
        for current_set, pattern_set in enumerate(all_data):
            for pattern in pattern_set[1]:
                if randint(0, 3) == 0:
                    neural.add_test(list(convert_pattern(pattern)), get_output(current_set, num_sets))
                else:
                    neural.add_train(list(convert_pattern(pattern)), get_output(current_set, num_sets))

        # for i in range(25):
        #     neural.add_train(numpy.random.rand(config.SIZE), [0] * num_sets)
        # for i in range(100):
        #     neural.add_test(numpy.random.rand(config.SIZE), [0] * num_sets)
        neural.train(ite)
        print(neural.log())

        def cb(sign):
            answer = neural.use(sign)[0]
            max_index, max_value = max(enumerate(answer), key=operator.itemgetter(1))
            print(all_data[max_index][0], max_value)

        Board(cb).capture(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str,
                        help="save pattern with given name")
    parser.add_argument("--train", type=int,
                        help="train network for given number of iterations")
    parser.add_argument("--stats",
                        help="show database statistics", action="store_true")

    args = parser.parse_args()
    if args.save:
        save_signature(args.save)
    elif args.train:
        test_network(args.train)
    elif args.stats:
        for val in load_all():
            print(val[0], len(val[1]))
