from data.normalize import differentiate, normalize, quantize
from data.pattern import load_pattern


def get_element(data, index):
    result = []
    for tup in data:
        result.append(tup[index])
    return result


def convert_pattern(data):
    x = quantize(get_element(data, 0), 100)
    y = quantize(get_element(data, 1), 100)
    t = quantize(get_element(data, 2), 100)
    vx = differentiate(x, t)
    vy = differentiate(y, t)
    ax = differentiate(vx, t)
    ay = differentiate(vy, t)

    for val in map(normalize, (x, y, vx, vy, ax, ay)):
        yield val
