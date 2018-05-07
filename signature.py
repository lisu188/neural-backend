import argparse
from random import randint

import tensorflow

from data.config import NUM_VECTORS, SIZE
from data.conversion import convert_pattern
from data.pattern import load_all
from engine.network import Network
from engine.runner import get_output


def test_network(ite):
    with tensorflow.Session() as sess:
        all_data = load_all()
        num_sets = len(all_data)
        neural = Network(sess, SIZE * NUM_VECTORS, num_sets)
        # neural.add_test([0] * SIZE * NUM_VECTORS, [0] * num_sets)
        for current_set, pattern_set in enumerate(all_data):
            for pattern in pattern_set[1]:
                if randint(0, 3) == 0:
                    neural.add_test(list(convert_pattern(pattern)), get_output(current_set, num_sets))
                else:
                    neural.add_train(list(convert_pattern(pattern)), get_output(current_set, num_sets))

        # for i in range(25):
        #     neural.add_train(numpy.random.rand(SIZE * NUM_VECTORS), [0] * num_sets)
        # for i in range(100):
        #     neural.add_test(numpy.random.rand(config.SIZE), [0] * num_sets)
        neural.train(ite)
        print(neural.log())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=int,
                        help="train network for given number of iterations")
    parser.add_argument("--stats",
                        help="show database statistics", action="store_true")

    args = parser.parse_args()

    if args.train:
        test_network(args.train)
    elif args.stats:
        for val in load_all():
            print(val[0], len(val[1]))
