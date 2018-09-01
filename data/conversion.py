from functools import partial

from data import config
from data.operations import differentiate_smooth, normalize, quantize


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
    def _composed(x):
        return a(b(x))

    return _composed


quntizer_op = partial(quantize, config.VECTOR_SIZE)
composed = compose(normalize, quntizer_op)


def convert_pattern(raw_data):
    sorted_data = sorted(raw_data, key=lambda ob: ob['Timestamp'])

    t = get_value_list(sorted_data, 'Timestamp')
    x = get_value_list(sorted_data, 'X')
    y = get_value_list(sorted_data, 'Y')
    f = get_value_list(sorted_data, 'Force')
    az = get_value_list(sorted_data, 'AzimuthAngle')
    al = get_value_list(sorted_data, 'AltitudeAngle')

    vx = differentiate_smooth(x, t)
    vy = differentiate_smooth(y, t)
    vf = differentiate_smooth(f, t)
    vaz = differentiate_smooth(az, t)
    val = differentiate_smooth(al, t)

    return {
        "vx": composed(vx),
        "vy": composed(vy),
        "vf": composed(vf),
        "vaz": composed(vaz),
        "val": composed(val),
    }


def convert_pattern_flat(raw_data):
    bean = convert_pattern(raw_data)
    for val in map(composed, (bean['vx'], bean['vy'], bean['vf'], bean['vaz'], bean['val'])):
        yield from val


def convert_pattern_split(raw_data):
    bean = convert_pattern(raw_data)
    for val in map(composed, (bean['vx'], bean['vy'], bean['vf'], bean['vaz'], bean['val'])):
        yield val
