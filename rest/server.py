#!flask/bin/python
import operator
from random import randint

import tensorflow
from flask import Flask, jsonify, request

from data.conversion import convert_pattern
from data.pattern import load_all
from engine.network import Network, series
from engine.runner import get_output


class NeuralRestEnginge:
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
        self.neural = [Network(self.session, self.num_sets) for i in series]
        # TODO: move to network
        for current_set, pattern_set in enumerate(self.all_data):
            for pattern in pattern_set[1]:
                for index, dataset in enumerate(convert_pattern(pattern)):
                    if randint(0, 3) == 0:
                        self.neural[index].add_test(dataset, get_output(current_set, self.num_sets))
                    else:
                        self.neural[index].add_train(dataset, get_output(current_set, self.num_sets))

    def train(self, steps):
        res = {}
        for index,name in enumerate(series):
            self.neural[index].train(steps)
            res[name] = self.neural[index].log()
        return res

    def run_server(self):
        app = Flask("neural")

        @app.route('/train/<int:steps>', methods=["POST"])
        def train(steps):
            return jsonify(self.train(steps))

        @app.route('/use', methods=["POST"])
        def use():
            result = {}
            for index, dataset in enumerate(convert_pattern(request.get_json(force=True))):
                answer = self.neural[index].use(dataset)[0]
                max_index, max_value = max(enumerate(answer), key=operator.itemgetter(1))
                result[series[index]] = {"pattern": self.all_data[max_index][0], "confidence": max_value.item()}
            return jsonify(result)

        @app.route('/reload', methods=["POST"])
        def reload():
            self.reload()
            ret = {'signatures': {}}
            for data in self.all_data:
                ret['signatures'][data[0]] = len(data[1])
            return jsonify(ret)

        app.run(debug=True)
