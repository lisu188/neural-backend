from data.normalize import differentiate, normalize, quantize
from data.pattern import load_pattern


def get_element(data, index):
    result = []
    for tup in data:
        result.append(tup[index])
    return result


def convert_pattern(data):
    x = quantize(get_element(data, 0), 1000)
    y = quantize(get_element(data, 1), 1000)
    t = quantize(get_element(data, 2), 1000)
    vx = differentiate(x, t)
    vy = differentiate(y, t)
    ax = differentiate(vx, t)
    ay = differentiate(vy, t)

    for val in map(normalize, (x, y, vx, vy, ax, ay)):
        yield from val


if __name__ == '__main__':
    def chart(name, *args):
        import pygal
        line_chart = pygal.Line(show_dots=False)
        for set in args:
            line_chart.add('', set)
        line_chart.render_to_file(name + '.svg')


    i = 0
    pattern_name = "counterclockwise"
    for pattern in load_pattern(pattern_name):
        chart(pattern_name + str(i), *convert_pattern(pattern))
        i = i + 1
