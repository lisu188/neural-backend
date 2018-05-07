from functools import partial

from data import config
from data.normalize import differentiate, normalize, quantize


def get_element(data, index):
    result = []
    for tup in data:
        result.append(tup[index])
    return result


def get_value_list(data, name):
    ret = []
    for ob in data:
        ret.append(ob[name])
    return ret


def compose(a, b):
    def composed(x):
        return a(b(x))

    return composed;


def convert_pattern(raw_data):
    sorted_data = sorted(raw_data, key=lambda ob: ob['Timestamp'])

    t = get_value_list(sorted_data, 'Timestamp')
    x = get_value_list(sorted_data, 'X')
    y = get_value_list(sorted_data, 'Y')
    f = get_value_list(sorted_data, 'Force')
    az = get_value_list(sorted_data, 'AzimuthAngle')
    al = get_value_list(sorted_data, 'AltitudeAngle')

    vx = differentiate(x, t)
    vy = differentiate(y, t)
    vf = differentiate(f, t)
    vaz = differentiate(az, t)
    val = differentiate(al, t)

    quntizer_op = partial(quantize, config.SIZE)

    composed = compose(normalize, quntizer_op)
    for val in map(composed, (vx, vy, vf, vaz, val)):
        yield from val
