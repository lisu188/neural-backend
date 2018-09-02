from random import random

from data.config import VECTOR_SIZE, VECTOR_COUNT, HIDDEN, RANDOM_TRAIN, RANDOM_TEST, NOISE_LEVEL
from data.conversion import convert_pattern_split
from data.pattern import load_all
from engine.network import Network


def get_output(index, num_sets):
    result = [0] * num_sets
    result[index] = 1
    return result


def build_random():
    return list(map(lambda x: list(map(lambda x: (random() - 0.5), range(VECTOR_SIZE))), range(VECTOR_COUNT)))


def count_train(data):
    total = 0
    for current_set_name, pattern_set in data.items():
        total += len(pattern_set['train'])
    return total


def count_test(data):
    total = 0
    for current_set_name, pattern_set in data.items():
        total += len(pattern_set['test'])
    return total


def noisify(pattern):
    new_pattern = []
    for serie in pattern:
        new_serie = []
        new_pattern.append(new_serie)
        for val in serie:
            new_serie.append(val + ((random() - 0.5) * NOISE_LEVEL))
    return new_pattern


def build_network(session):
    all_data = load_all()
    num_sets = len(all_data)
    neural = Network(session, [VECTOR_COUNT, VECTOR_SIZE], num_sets, HIDDEN)

    for current_set_name, pattern_set in all_data.items():
        current_set_id = list(all_data.keys()).index(current_set_name)
        output = get_output(current_set_id, num_sets)
        for pattern in pattern_set['test']:
            pattern = list(convert_pattern_split(pattern['data']))
            neural.add_test(pattern, output)
        for pattern in pattern_set['train']:
            pattern = list(convert_pattern_split(pattern['data']))
            neural.add_train(pattern, output)

    train_ = count_train(all_data) * RANDOM_TRAIN // 100
    test_ = count_test(all_data) * RANDOM_TEST // 100
    for i in range(train_):
        neural.add_train(build_random(), [0] * num_sets)
    for i in range(test_):
        neural.add_test(build_random(), [0] * num_sets)

    return neural
