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


# TODO: check if timebase is sorted?
def quantize_base(dest_len, data, base):
    assert len(data) == len(base)

    base = normalize(base, 0)
    new_base = list(map(lambda x: x * 1 / (dest_len - 1), range(dest_len)))

    new_data = []
    current_index = 0
    for new_index in range(dest_len):
        if new_base[new_index] > base[current_index + 1]:
            current_index += 1
        if new_base[new_index] == base[current_index]:
            new_data.append(data[current_index])
        else:
            base_diff = base[current_index + 1] - base[current_index]

            data_diff = data[current_index + 1] - data[current_index]

            new_base_diff = new_base[new_index] - base[current_index]

            ratio = new_base_diff / base_diff

            new_data.append(data[current_index] + data_diff * ratio)
    return new_data


def normalize(data, shift=-0.5):
    local_min = min(data)
    local_max = max(data) - local_min

    def div_max(x):
        if local_max == 0:
            return 0
        return x / local_max

    return list(map(lambda x: (x + shift), map(div_max, map(lambda x: x - local_min, data))))


def get_value_list(data, name):
    ret = []
    for ob in data:
        ret.append(ob[name])
    return ret


def smooth(data):
    return savitzky_golay(data, 13, 3)


def differentiate(x, t):
    assert (len(x) == len(t))
    result = []
    for i in range(len(x) - 1):
        result.append((x[i + 1] - x[i]) / (t[i + 1] - t[i]))
    return result


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
        "f": array_rms(get_value_list(patterns, 'f')),
        "az": array_rms(get_value_list(patterns, 'az')),
        "al": array_rms(get_value_list(patterns, 'al'))
    }


def average_pattern(patterns):
    return {
        "vx": average_array(get_value_list(patterns, 'vx')),
        "vy": average_array(get_value_list(patterns, 'vy')),
        "f": average_array(get_value_list(patterns, 'f')),
        "az": average_array(get_value_list(patterns, 'az')),
        "al": average_array(get_value_list(patterns, 'al'))
    }


if __name__ == '__main__':
    print(quantize(100, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    print(normalize([10, 20, 30, 40, 50]))
    print(differentiate([2, 4, 6, 8, 10], [1, 2, 3, 4, 5]))
    print(quantize_base(10, [1, 3, 5], [0, 1, 100]))
