import operator
import os

import tensorflow
from flask import Flask, jsonify, request

from data.conversion import convert_pattern
from data.operations import average_pattern, rms
from data.pattern import load_all
from engine.runner import build_network


class NeuralRestEngine:
    def __init__(self):
        self.session = None
        self.all_data = None
        self.num_sets = None
        self.neural = None
        self.reload()

    def reload(self):
        if self.session:
            self.session.close()
        self.session = tensorflow.Session()
        self.all_data = load_all()
        self.num_sets = len(self.all_data)
        self.neural = build_network(self.session)
        return self.neural.log()

    def train(self, steps):
        self.neural.train(steps)
        return self.neural.log()

    def run_server(self):
        app = Flask("neural")

        @app.route('/train/<int:steps>', methods=["POST"])
        def train(steps):
            return jsonify(self.train(steps))

        @app.route('/use', methods=["POST"])
        def use():
            answer = self.neural.use(request.get_json(force=True))[0]
            max_index, max_value = max(enumerate(answer), key=operator.itemgetter(1))
            return jsonify({"pattern": self.all_data[max_index][0], "confidence": max_value.item()})

        @app.route('/reload', methods=["POST"])
        def reload():
            return jsonify(self.reload())

        @app.route('/stats', methods=['GET'])
        def stats():
            ret = {'signatures': {}}
            for data in self.all_data:
                ret['signatures'][data[0]] = {}
                ret['signatures'][data[0]]['train'] = len(data[1]['train'])
                ret['signatures'][data[0]]['test'] = len(data[1]['test'])
            return jsonify(ret)

        @app.route('/api', methods=['GET'])
        def api():
            output = []
            for rule in app.url_map.iter_rules():
                methods = list(filter(lambda x: x != 'HEAD' and x != 'OPTIONS', rule.methods))
                output.append({"endpoint": str(rule), "methods": methods})

            return jsonify(output)

        @app.route('/chart/<name>/<type>/<int:id>', methods=['GET'])
        def chart(name, type, id):
            import pygal
            line_chart = pygal.Line(show_dots=False)
            y = load_all()[name]
            for key, val in convert_pattern(y[type][id]).items():
                line_chart.add(key, list(val))
            return line_chart.render_response()

        @app.route('/chart/<name>/<type>', methods=['GET'])
        def avg_chart(name, type):
            import pygal
            line_chart = pygal.Line(show_dots=False)
            y = load_all()[name]
            for key, val in average_pattern(list(map(convert_pattern, y[type]))).items():
                line_chart.add(key, list(val))
            return line_chart.render_response()

        @app.route('/chart/<name>', methods=['GET'])
        def avg_chart_all(name):
            import pygal
            line_chart = pygal.Line(show_dots=False)
            y = load_all()[name]
            for key, val in average_pattern(list(map(convert_pattern, y['train'] + y['test']))).items():
                line_chart.add(key, list(val))
            return line_chart.render_response()

        @app.route('/signature/<name>/<type>/<int:id>', methods=['GET'])
        def signature(name, type, id):
            import pygal
            xy_chart = pygal.XY(stroke=False)
            y = load_all()[name]
            xy_chart.add(name, list(map(lambda point: (point['X'], -point['Y']), y[type][id])))
            return xy_chart.render_response()

        @app.route('/rmsd', methods=['GET'])
        def rmsd():
            all_rms = {}
            for key, val in load_all().items():
                all_patterns = val['test'] + val['train']
                rms_values = rms(list(map(lambda x: convert_pattern(x['data']), all_patterns)))
                map_file_to_rms = {}
                for i in range(len(all_patterns)):
                    map_file_to_rms[all_patterns[i]['file']] = {}
                    for param in ['vx', 'vy', 'vf', 'vaz', 'val']:
                        map_file_to_rms[all_patterns[i]['file']][param] = rms_values[param][i]
                all_rms[key] = map_file_to_rms
            return jsonify(all_rms)

        if 'PORT' in os.environ:
            port = int(os.environ['PORT'])
        else:
            port = 5000

        app.run(host='0.0.0.0', port=port, debug=True)
