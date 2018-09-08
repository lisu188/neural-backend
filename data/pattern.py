import json
import os

from data.operations import shuffled

PATH = "patterns"
EXCLUDED = []
_PATTERNS = None


def load_all():
    global _PATTERNS
    if _PATTERNS is None:
        _PATTERNS = {}
        for dir in filter(lambda x: x not in EXCLUDED, os.listdir(PATH)):
            _PATTERNS[dir] = []
            for file in shuffled(os.listdir(os.path.join(PATH, dir))):
                _PATTERNS[dir].append(
                    {"file": file, "data": json.loads(open(os.path.join(PATH, dir, file)).read())})
    else:
        return _PATTERNS


def get_num_classes():
    return len(list(filter(lambda x: x not in EXCLUDED, os.listdir(PATH))))
