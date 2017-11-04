from data.conversion import convert_pattern
from data.pattern import load_pattern, load_all
from engine.network import series

if __name__ == '__main__':

    def chart(name, args):
        import pygal
        line_chart = pygal.Line(show_dots=False)
        for ind, set in enumerate(args):
            line_chart.add(series[ind], set)
        line_chart.render_to_file("charts/" + name + '.svg')


    for pattern in load_all():
        for index, dataset in enumerate(pattern[1]):
            chart(pattern[0] + str(index), convert_pattern(dataset))
