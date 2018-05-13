import math


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(np.asarray(y[1:half_window + 1][::-1]) - y[0])
    lastvals = y[-1] + np.abs(np.asarray(y[-half_window - 1:-1][::-1]) - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def quantize(dest_len, data):
    src_len = len(data)
    ratio = (src_len - 1) / dest_len

    result = []
    for i in range(dest_len - 1):
        source_index = i * ratio
        upper_index = math.ceil(source_index)
        lower_index = math.floor(source_index)
        distance_factor = math.modf(source_index)[0]
        result.append(data[lower_index] + (data[upper_index] - data[lower_index]) * distance_factor)
    result.append(data[src_len - 1])
    return result


def normalize(data):
    local_min = min(data)
    local_max = max(data) - local_min
    return list(map(lambda x: (x - 0.5) * 2, map(lambda x: x / local_max, map(lambda x: x - local_min, data))))


def get_value_list(data, name):
    ret = []
    for ob in data:
        ret.append(ob[name])
    return ret


def differentiate(dx, dt):
    assert (len(dx) == len(dt))
    result = []
    for i in range(len(dx) - 1):
        result.append((dx[i + 1] - dx[i]) / (dt[i + 1] - dt[i]))
    return savitzky_golay(result, 51, 3)


def average_array(arrays):
    # TODO: assert all arrays same length
    result = []
    for i in range(len(arrays[0])):
        result.append(sum(map(lambda x: x[i], arrays)) / len(arrays))
    return result


def avg(*args):
    return sum(args) / len(args)


def arr_rms(array1, array2):
    return math.sqrt(sum(map(lambda val: math.pow(val[0] - val[1], 2), zip(array1, array2))))


def array_rms(arrays):
    avg_arr = average_array(arrays)
    return list(map(lambda val: arr_rms(val, avg_arr), arrays))


def rms(patterns):
    return {
        "vx": array_rms(get_value_list(patterns, 'vx')),
        "vy": array_rms(get_value_list(patterns, 'vy')),
        "vf": array_rms(get_value_list(patterns, 'vf')),
        "vaz": array_rms(get_value_list(patterns, 'vaz')),
        "val": array_rms(get_value_list(patterns, 'val'))
    }


def average_pattern(patterns):
    return {
        "vx": average_array(get_value_list(patterns, 'vx')),
        "vy": average_array(get_value_list(patterns, 'vy')),
        "vf": average_array(get_value_list(patterns, 'vf')),
        "vaz": average_array(get_value_list(patterns, 'vaz')),
        "val": average_array(get_value_list(patterns, 'val'))
    }


if __name__ == '__main__':
    print(quantize([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 100))
    print(normalize([10, 20, 30, 40, 50]))
    print(differentiate([2, 4, 6, 8, 10], [1, 2, 3, 4, 5]))
