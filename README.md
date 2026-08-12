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

The original deployment configuration starts the same server entry point through the included `Procfile`.

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

`requirements.txt` records the historical top-level dependencies, but versions are not pinned and modern TensorFlow releases are not API-compatible with this code. Reproducing the original environment therefore requires an appropriately old Python/TensorFlow stack or a migration of the session-based code.
