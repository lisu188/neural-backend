# Neural Backend

A historical machine-learning experiment combining **TensorFlow**, **Flask**, NumPy and lightweight SVG charting to train and inspect a neural classifier over stored input patterns.

This repository is kept public as an engineering archive. It demonstrates an older TensorFlow 1.x-style workflow (`tensorflow.Session`) and should **not** be read as a recommendation for a modern production ML stack.

## What it demonstrates

- neural-network construction and training behind a small Python application layer
- a Flask REST API for training, inference, reloads, statistics and diagnostics
- thread-safe access to a shared model session
- pattern conversion, caching and aggregate statistics
- chart endpoints for inspecting stored signatures and derived vectors
- CLI entry points for training, running multiple model experiments, statistics and serving the API

## Main entry point

```bash
python3 signature.py --server
```

Other modes exposed by the CLI include:

```bash
python3 signature.py --train 1000
python3 signature.py --stats
python3 signature.py --models 4 --train 1000
```

The server binds to `127.0.0.1` by default. The API has no built-in authentication and can train or reload the shared model and return stored patterns, so keep it local unless an authenticated reverse proxy restricts access. Remote deployments must explicitly choose a bind address with `NEURAL_HOST` (for example, `NEURAL_HOST=0.0.0.0` behind that proxy); `PORT` still selects the port, defaulting to 5000.

The original deployment configuration starts the same server entry point through the included `Procfile`; remote hosting now requires that explicit bind configuration.

## Structure

- `engine/` — network construction and execution
- `rest/` — Flask API
- `data/` — pattern loading, conversion, caching and statistics
- `patterns/` — pattern data used by the experiment
- `gui/` — supporting visualization/UI code
- `signature.py` — CLI and application entry point

## Status

**Legacy / experimental.** The project predates modern TensorFlow APIs and dependency pinning practices. I keep it public to show the evolution of my ML experimentation and backend work rather than as maintained production software.

## Running today

`requirements.txt` records the top-level dependencies, including the previously
missing `requests` import. Installing its unconstrained TensorFlow on a current
Python interpreter does not reproduce this session-based application.

`requirements-legacy.txt` is a candidate compatibility profile for CPython 3.7
on Linux x86-64. It selects TensorFlow 1.15.5 and NumPy 1.18.5 and constrains the
older Flask/Protobuf dependencies. It is not a complete transitive lock or a
claim that installation and training have passed. The original `runtime.txt`
records Python 3.6.5; it remains historical deployment metadata.

In an isolated CPython 3.7 environment:

```bash
python3.7 -m venv .venv-legacy
.venv-legacy/bin/python -m pip install 'pip==24.0' 'setuptools==59.6.0' 'wheel==0.37.1'
.venv-legacy/bin/python -m pip install -r requirements-legacy.txt
.venv-legacy/bin/python -m pip check
.venv-legacy/bin/python -c 'import tensorflow as tf; assert tf.__version__ == "1.15.5"; print(tf.Session)'
.venv-legacy/bin/python signature.py --train 1
.venv-legacy/bin/python -m pip freeze > legacy-resolved.txt
```

Keep the resolved versions and actual smoke results together before describing
that environment as reproduced. A Python 3/TensorFlow 2 migration remains
separate work. The versions above are supported by their historical package
metadata: [TensorFlow 1.15.5](https://pypi.org/project/tensorflow/1.15.5/),
[NumPy 1.18.5](https://pypi.org/project/numpy/1.18.5/),
[Flask 1.1.4](https://pypi.org/project/Flask/1.1.4/), and
[Requests 2.27.1](https://pypi.org/project/requests/2.27.1/).

### Pattern source and offline checks

The default data source is now the checked-in `patterns/` directory. Set
`NEURAL_PATTERN_DIR` to use another local directory with the same class/JSON
layout. Loading local patterns does not contact the former HTTP service.
`NEURAL_PATTERN_URL` explicitly selects an HTTPS ZIP endpoint instead; downloads
have a timeout and size limits, and unsafe archive paths are rejected before
extraction. `--upload` also requires this explicit HTTPS endpoint.

The source/configuration checks need only a current Python interpreter:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q data engine rest gui signature.py
```

`--help` and `--stats` work without TensorFlow, Flask or site packages. Statistics
count each local class and use the engine's existing floored test percentage;
training imports TensorFlow only when requested, and the server imports its HTTP
stack only for `--server`. Multiple-model training uses a managed process pool.

The checks include real `python -S signature.py --help/--stats` subprocesses,
local loading, remote source admission, archive traversal rejection and server
binding. Pool orchestration is tested separately from training. None of these
checks instantiate or validate TensorFlow.
