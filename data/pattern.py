import json
import os
from os.path import basename, splitext


def load_all():
    patterns = []
    for file in os.listdir("patterns"):
        patterns.append((splitext(basename(file))[0], json.loads(open("patterns/" + file).read())))
    return patterns


def save_pattern(name, data):
    open("patterns/" + name + ".ptn", "w").write(json.dumps(data, indent=4, sort_keys=True))


def add_pattern(name, data):
    previous = load_pattern(name)
    save_pattern(name, previous + data)


def load_pattern(name):
    try:
        return json.loads(open("patterns/" + name + ".ptn").read())
    except IOError:
        return []
