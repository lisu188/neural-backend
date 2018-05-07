import json
import os

PATH = "pattern-database"
EXCLUDED = [".git"]


def load_all():
    patterns = {}
    for dir in filter(lambda x: x not in EXCLUDED, os.listdir(PATH)):
        patterns[dir] = []
        for file in os.listdir(os.path.join(PATH, dir)):
            patterns[dir].append(json.loads(open(os.path.join(PATH, dir, file)).read()))
    return patterns.items()


def get_num_classes():
    return len(list(filter(lambda x: x not in EXCLUDED, os.listdir(PATH))))
