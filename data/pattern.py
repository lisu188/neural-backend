import json
import os
import shutil
import zipfile
from io import BytesIO

import requests

from data.cache import cached
from data.operations import shuffled

PATH = "patterns"
PATH_PATTERNS = 'tmp_patterns'

URL = "http://middleware.northeurope.azurecontainer.io/api/Signatures"

_PATTERNS = None


def load_from_disk(path):
    patterns = {}
    for dir in os.listdir(path):
        patterns[dir] = []
        for file in shuffled(os.listdir(os.path.join(path, dir))):
            patterns[dir].append(
                {"file": file, "data": json.loads(open(os.path.join(path, dir, file)).read())})
    return patterns


@cached
def load_all():
    r = requests.get(URL, stream=True)
    z = zipfile.ZipFile(BytesIO(r.content))

    if os.path.exists(PATH_PATTERNS):
        shutil.rmtree(PATH_PATTERNS)

    z.extractall(PATH_PATTERNS)
    return load_from_disk(PATH_PATTERNS)


def get_num_classes():
    return len(load_all())


def upload_all():
    for name, patterns in load_from_disk(PATH).items():
        for pattern in patterns:
            requests.post(URL, json={'className': name, 'data': json.dumps(pattern['data'])})
