from data import config
from data.operations import quantize_base, normalize, differentiate, smooth


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

    vx = quantize_base(config.VECTOR_SIZE, vx, t[0:-1])
    vy = quantize_base(config.VECTOR_SIZE, vy, t[0:-1])
    f = quantize_base(config.VECTOR_SIZE, f, t)
    az = quantize_base(config.VECTOR_SIZE, az, t)
    al = quantize_base(config.VECTOR_SIZE, al, t)

    vx = normalize(vx)
    vy = normalize(vy)
    f = normalize(f)
    az = normalize(az)
    al = normalize(al)

    vx = smooth(vx)
    vy = smooth(vy)
    f = smooth(f)
    az = smooth(az)
    al = smooth(al)

    return {
        "vx": vx,
        "vy": vy,
        "f": f,
        "az": az,
        "al": al,
    }


def convert_pattern_split(raw_data):
    bean = convert_pattern(raw_data)
    for val in (bean['vx'], bean['vy'], bean['f'], bean['az'], bean['al']):
        yield val
