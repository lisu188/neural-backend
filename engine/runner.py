from random import randint

from data.config import SIZE, NUM_VECTORS
from data.conversion import convert_pattern
from data.pattern import load_all
from engine.network import Network


def get_output(index, num_sets):
    result = [0] * num_sets
    result[index] = 1
    return result


def build_network(session):
    all_data = load_all()
    num_sets = len(all_data)
    neural = Network(session, SIZE * NUM_VECTORS, num_sets)

    for current_set, pattern_set in enumerate(all_data):
        for pattern in pattern_set[1]:
            if randint(0, 3) == 0:
                neural.add_test(list(convert_pattern(pattern)), get_output(current_set, num_sets))
            else:
                neural.add_train(list(convert_pattern(pattern)), get_output(current_set, num_sets))

    return neural
