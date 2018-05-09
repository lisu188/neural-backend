import operator
import os

import tensorflow
from flask import Flask, jsonify, request

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
            self.reload()
            ret = {'signatures': {}}
            for data in self.all_data:
                ret['signatures'][data[0]] = len(data[1])
            return jsonify(ret)

        # if 'PORT' in os.environ:
        #     port = int(os.environ['PORT'])
        # else:
        #     port = 5000

        app.run(host='0.0.0.0')
