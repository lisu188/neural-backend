import argparse
from multiprocessing import Pool

from data.config import TEST_PERCENT
from data.pattern import load_all, upload_all


def test_network(ite):
    import tensorflow
    from engine.runner import build_network

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
    parser.add_argument("--upload",
                        help="upload local patterns to database", action="store_true")
    args = parser.parse_args()

    if args.train:
        if args.models:
            with Pool(args.models) as pool:
                pool.map(test_network, [args.train] * args.models)
        else:
            test_network(args.train)
    elif args.stats:
        for key, val in load_all().items():
            test_count = TEST_PERCENT * len(val) // 100
            print(key, len(val))
            print('\t', 'train', len(val) - test_count)
            print('\t', 'test', test_count)
    elif args.server:
        from rest.server import NeuralRestEngine

        NeuralRestEngine().run_server()
    elif args.upload:
        upload_all()

