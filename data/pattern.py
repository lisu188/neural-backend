import json
import os
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from data.cache import cached
from data.operations import shuffled

PATH = "patterns"


def load_from_disk(path):
    patterns = {}
    # Class indices and the train/test split follow the legacy directory order
    # and shuffled file order. Reproduction must not silently remap classes.
    for name in os.listdir(path):
        directory = Path(path) / name
        if not directory.is_dir():
            continue
        records = []
        for filename in shuffled(os.listdir(str(directory))):
            file = directory / filename
            if not file.is_file() or file.suffix != '.json':
                continue
            with file.open(encoding="utf-8") as source:
                records.append({"file": file.name, "data": json.load(source)})
        if records:
            patterns[directory.name] = records
    if not patterns:
        raise ValueError("No class directories containing JSON patterns in " + str(path))
    return patterns


def remote_url():
    url = os.environ.get("NEURAL_PATTERN_URL", "")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("NEURAL_PATTERN_URL must be an explicit HTTPS endpoint without credentials")
    return url


@cached
def load_all():
    # A checkout must be usable without the historical external HTTP service.
    if not os.environ.get("NEURAL_PATTERN_URL"):
        return load_from_disk(os.environ.get("NEURAL_PATTERN_DIR", PATH))

    import requests
    with requests.get(remote_url(), stream=True, timeout=30, allow_redirects=False) as response:
        response.raise_for_status()
        if 300 <= response.status_code < 400:
            raise ValueError("Pattern endpoint redirects are not accepted")
        payload = bytearray()
        for chunk in response.iter_content(65536):
            payload.extend(chunk)
            if len(payload) > 32 * 1024 * 1024:
                raise ValueError("Pattern archive download exceeds 32 MiB")
    # Keep extraction isolated and reject traversal before writing any member.
    with zipfile.ZipFile(BytesIO(payload)) as archive, tempfile.TemporaryDirectory() as directory:
        total = 0
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise ValueError("Unsafe pattern archive member")
            total += member.file_size
            if total > 64 * 1024 * 1024:
                raise ValueError("Expanded pattern archive exceeds 64 MiB")
        archive.extractall(directory)
        return load_from_disk(directory)


def get_num_classes():
    return len(load_all())


def upload_all():
    import requests
    url = remote_url()
    for name, patterns in load_from_disk(os.environ.get("NEURAL_PATTERN_DIR", PATH)).items():
        for pattern in patterns:
            response = requests.post(url, json={"className": name, "data": json.dumps(pattern["data"])},
                                     timeout=30, allow_redirects=False)
            response.raise_for_status()
            if 300 <= response.status_code < 400:
                raise ValueError("Pattern endpoint redirects are not accepted")
