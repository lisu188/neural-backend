import json
import os
from os.path import basename, splitext

from gui.board import Board

PATH = "pattern_database"


def load_all():
    patterns = []
    for file in os.listdir(PATH):
        split = splitext(basename(file))
        if split[1] == ".ptn":
            patterns.append((split[0], json.loads(open(PATH + "/" + file).read())))
    return patterns


def save_pattern(name, data):
    open(PATH + "/" + name + ".ptn", "w").write(json.dumps(data, indent=4, sort_keys=True))


def add_pattern(name, data):
    previous = load_pattern(name)
    save_pattern(name, previous + data)


def load_pattern(name):
    try:
        return json.loads(open(PATH + "/" + name + ".ptn").read())
    except IOError:
        return []
