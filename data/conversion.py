from data import config
from data.config import VECTORS
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
    # sort by time
    sorted_data = sorted(raw_data, key=lambda ob: ob['Timestamp'])

    t = get_value_list(sorted_data, 'Timestamp')  # get time vector
    x = get_value_list(sorted_data, 'X')  # get X vector
    y = get_value_list(sorted_data, 'Y')  # get Y vector
    f = get_value_list(sorted_data, 'Force')  # get Force vector
    az = get_value_list(sorted_data, 'AzimuthAngle')  # get azimuth vector
    al = get_value_list(sorted_data, 'AltitudeAngle')  # get altitude angle vector

    # calculate speed according to x and y
    vx = differentiate(x, t)
    vy = differentiate(y, t)

    # strech/shorten vector (size less than one becasue of differentation implementation
    vx = quantize_base(config.VECTOR_SIZE, vx,
                       t[0:-1])
    vy = quantize_base(config.VECTOR_SIZE, vy,
                       t[0:-1])
    f = quantize_base(config.VECTOR_SIZE, f, t)
    az = quantize_base(config.VECTOR_SIZE, az,
                       t)
    al = quantize_base(config.VECTOR_SIZE, al,
                       t)

    # apply savitsky golay
    vx = smooth(vx)
    vy = smooth(vy)
    f = smooth(f)
    az = smooth(az)
    al = smooth(al)

    # normalize to -0.5,0.5
    vx = normalize(vx)
    vy = normalize(vy)
    f = normalize(f)
    az = normalize(az)
    al = normalize(al)

    # pack values
    return {
        "vx": vx,
        "vy": vy,
        "f": f,
        "az": az,
        "al": al,
    }


def convert_pattern_split(raw_data):
    bean = convert_pattern(raw_data)
    for val in (map(lambda val: bean[val], VECTORS)):
        yield val
