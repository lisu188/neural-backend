from data.config import SIZE, NUM_VECTORS
from data.conversion import convert_pattern_flat
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

    for current_set_name, pattern_set in all_data.items():
        current_set_id = list(all_data.keys()).index(current_set_name)
        for pattern in pattern_set['test']:
            neural.add_test(list(convert_pattern_flat(pattern)), get_output(current_set_id, num_sets))
        for pattern in pattern_set['train']:
            neural.add_train(list(convert_pattern_flat(pattern)), get_output(current_set_id, num_sets))
    return neural
