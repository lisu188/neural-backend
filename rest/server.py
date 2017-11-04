#!flask/bin/python
import operator
from random import randint

import tensorflow
from flask import Flask, jsonify, request

from data.conversion import convert_pattern
from data.pattern import load_all
from engine.network import Network
from engine.runner import get_output


class NeuralRestEnginge:
    def __init__(self):
        self.session = tensorflow.Session()
        self.all_data = load_all()
        self.num_sets = len(self.all_data)
        self.neural = Network(self.session, self.num_sets)
        # TODO: move to network
        for current_set, pattern_set in enumerate(self.all_data):
            for pattern in pattern_set[1]:
                if randint(0, 3) == 0:
                    self.neural.add_test(list(convert_pattern(pattern)), get_output(current_set, self.num_sets))
                else:
                    self.neural.add_train(list(convert_pattern(pattern)), get_output(current_set, self.num_sets))

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

        app.run(debug=True)
