from random import random

from data.config import VECTOR_SIZE, VECTOR_COUNT, HIDDEN, RANDOM_SETS, NOISE_LEVEL, TEST_PERCENT
from data.conversion import convert_pattern_split
from data.pattern import load_all
from engine.network import Network


def get_output(index, num_sets):
    result = [0] * num_sets
    result[index] = 1
    return result


def build_random():
    return list(map(lambda x: list(map(lambda x: (random() - 0.5), range(VECTOR_SIZE))), range(VECTOR_COUNT)))


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

        converted_patterns = list(map(lambda pattern: list(convert_pattern_split(pattern['data'])), pattern_set))
        # converted_patterns = list(chain(converted_patterns, map(noisify, converted_patterns)))

        number_of_test_sets = TEST_PERCENT * len(converted_patterns) // 100

        # shuffle(converted_patterns)
        for pattern in converted_patterns[0:number_of_test_sets]:
            neural.add_test(pattern, output)
        for pattern in converted_patterns[number_of_test_sets:]:
            neural.add_train(pattern, output)

    random_train = RANDOM_SETS * TEST_PERCENT // 100
    for i in range(RANDOM_SETS - random_train):
        neural.add_train(build_random(), [0] * num_sets)
    for i in range(random_train):
        neural.add_test(build_random(), [0] * num_sets)

    return neural
