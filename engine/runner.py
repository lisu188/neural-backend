from random import randint

from data.conversion import convert_pattern
from data.pattern import load_all
from engine.network import Network


def get_output(index, num_sets):
    result = [0] * num_sets
    result[index] = 1
    return result


if __name__ == '__main__':
    all_data = load_all()
    num_sets = len(all_data)
    neural = Network()
    for current_set in range(num_sets):
        for pattern in all_data[current_set]:
            if randint(0,3) == 0:
                neural.add_test(list(convert_pattern(pattern)), get_output(current_set, num_sets))
            else:
                neural.add_train(list(convert_pattern(pattern)), get_output(current_set, num_sets))

    neural.train()
