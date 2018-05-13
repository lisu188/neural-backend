import json
import os

PATH = "patterns"
EXCLUDED = []


def load_all():
    patterns = {}
    for dir in filter(lambda x: x not in EXCLUDED, os.listdir(PATH)):
        patterns[dir] = {}
        patterns[dir]['test'] = []
        patterns[dir]['train'] = []
        for file in os.listdir(os.path.join(PATH, dir, 'test')):
            patterns[dir]['test'].append(
                {"file": file, "data": json.loads(open(os.path.join(PATH, dir, 'test', file)).read())})
        for file in os.listdir(os.path.join(PATH, dir, 'train')):
            patterns[dir]['train'].append(
                {"file": file, "data": json.loads(open(os.path.join(PATH, dir, 'train', file)).read())})
    return patterns


def get_num_classes():
    return len(list(filter(lambda x: x not in EXCLUDED, os.listdir(PATH))))
