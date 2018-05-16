import argparse

import tensorflow

from data.pattern import load_all
from engine.runner import build_network
from rest.server import NeuralRestEngine

from multiprocessing import Pool


def test_network(ite):
    with tensorflow.Session() as sess:
        neural = build_network(sess)
        neural.train(ite)
        print(neural.log())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=int,
                        help="train network for given number of iterations")
    parser.add_argument("--server",
                        help="start rest server", action="store_true")
    parser.add_argument("--stats",
                        help="show database statistics", action="store_true")
    parser.add_argument("--models", type=int,
                        help="how many models to train")
    args = parser.parse_args()

    if args.train:
        if args.models:
            pool = Pool(args.models)
            pool.map(test_network, [args.train] * args.models)
            pool.join()
        else:
            test_network(args.train)
    elif args.stats:
        for key, val in load_all().items():
            print(key, len(val['train']) + len(val['test']))
            print('\t', 'train', len(val['train']))
            print('\t', 'test', len(val['test']))
    elif args.server:
        NeuralRestEngine().run_server()

