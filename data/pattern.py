import json
import os

from data.operations import shuffled

PATH = "patterns"
EXCLUDED = []


def load_all():
    patterns = {}
    for dir in filter(lambda x: x not in EXCLUDED, os.listdir(PATH)):
        patterns[dir] = []
        for file in shuffled(os.listdir(os.path.join(PATH, dir))):
            patterns[dir].append(
                {"file": file, "data": json.loads(open(os.path.join(PATH, dir, file)).read())})
    return patterns


def get_num_classes():
    return len(list(filter(lambda x: x not in EXCLUDED, os.listdir(PATH))))
