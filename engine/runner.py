from random import randint

from data.conversion import convert_pattern
from data.pattern import load_all
from engine.network import Network


def get_output(index, num_sets):
    result = [0] * num_sets
    result[index] = 1
    return result


