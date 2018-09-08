import os
from threading import RLock, Thread

import tensorflow
from flask import Flask, jsonify, request

from data.config import VECTORS
from data.conversion import convert_pattern
from data.operations import average_pattern, rms, avg
from data.pattern import load_all
from engine.runner import build_network


def locking(lock):
    def _locking(f):
        def locked_f(*args, **kwargs):
            with lock:
                return f(*args, **kwargs)

        return locked_f

    return _locking


class NeuralRestEngine:
    def __init__(self):
        self.session = None
        self.num_sets = None
        self.neural = None
        self.lock = RLock()
        self.reload()

    def reload(self):
        if self.session:
            self.session.close()
        self.session = tensorflow.Session()
        self.num_sets = len(load_all())
        self.neural = build_network(self.session)
        return self.neural.log()

    def train(self, steps):
        self.neural.train(steps)

    def run_server(self):
        app = Flask("neural")

        @app.route('/train/<int:steps>', methods=["POST"], endpoint="train")
        def train(steps):
            def train_op():
                for i in range(steps):
                    with self.lock:
                        self.train(1)

            Thread(target=train_op).start()
            return '', 204

        @app.route('/use', methods=["POST"], endpoint="use")
        @locking(lock=self.lock)
        def use():
            answer = self.neural.use(request.get_json(force=True))[0]
            ret = {}
            for key, val in zip(list(load_all().keys()), answer):
                ret[str(key)] = float(val)
            return jsonify(ret)

        @app.route('/log', methods=["GET"], endpoint="log")
        @locking(lock=self.lock)
        def log():
            return jsonify(self.neural.log())

        @app.route('/reload', methods=["POST"], endpoint="reload")
        @locking(lock=self.lock)
        def reload():
            return jsonify(self.reload())

        @app.route('/stats', methods=['GET'])
        def stats():
            ret = {'signatures': {}}
            for data in load_all():
                ret['signatures'][data[0]] = len(data[1])
            return jsonify(ret)

        @app.route('/api', methods=['GET'])
        def api():
            output = []
            for rule in app.url_map.iter_rules():
                methods = list(filter(lambda x: x != 'HEAD' and x != 'OPTIONS', rule.methods))
                output.append({"endpoint": str(rule), "methods": methods})

            return jsonify(output)

        @app.route('/chart/<name>/<int:id>', methods=['GET'])
        def chart(name, id):
            import pygal
            line_chart = pygal.Line(show_dots=False)
            y = load_all()[name]
            for key, val in convert_pattern(y[id]['data']).items():
                line_chart.add(key, list(val))
            return line_chart.render_response()

        @app.route('/chart/<name>', methods=['GET'])
        def avg_chart(name):
            import pygal
            line_chart = pygal.Line(show_dots=False)
            y = load_all()[name]
            for key, val in average_pattern(
                    list(map(convert_pattern, list(map(lambda x: x['data'], y))))).items():
                line_chart.add(key, list(val))
            return line_chart.render_response()

        @app.route('/signature/<name>/<int:id>', methods=['GET'])
        def signature(name, id):
            import pygal
            xy_chart = pygal.XY(stroke=False)
            y = load_all()[name]
            xy_chart.add(name, list(map(lambda point: (point['X'], -point['Y']), y[id]['data'])))
            return xy_chart.render_response()

        @app.route('/rmsd', methods=['GET'])
        def rmsd():
            all_rms = {}
            for key, val in load_all().items():
                all_patterns = val
                rms_values = rms(list(map(lambda x: convert_pattern(x['data']), all_patterns)))
                map_file_to_rms = {}
                for i in range(len(all_patterns)):
                    map_file_to_rms[all_patterns[i]['file']] = {}
                    for param in VECTORS:
                        map_file_to_rms[all_patterns[i]['file']][param] = rms_values[param][i]
                    map_file_to_rms[all_patterns[i]['file']]['avg'] = avg(
                        *map_file_to_rms[all_patterns[i]['file']].values())
                all_rms[key] = sorted(list(map_file_to_rms.items()), key=lambda x: x[1]['avg'], reverse=True)
            return jsonify(all_rms)

        if 'PORT' in os.environ:
            port = int(os.environ['PORT'])
        else:
            port = 5000

        app.run(host='0.0.0.0', port=port)
