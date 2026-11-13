"""Real offline CLI subprocesses; no TensorFlow or site packages are loaded."""
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class OfflineCliTests(unittest.TestCase):
    def command(self, arguments, directory, environment):
        return subprocess.run([sys.executable, '-S', str(ROOT / 'signature.py'), *arguments],
                              cwd=directory, env=environment, capture_output=True,
                              text=True, timeout=10)

    def test_help_runs_without_site_packages_or_data(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.command(['--help'], directory, os.environ.copy())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--stats', result.stdout)
        self.assertIn('--server', result.stdout)

    def test_stats_counts_real_local_patterns_and_engine_split_without_tensorflow(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'patterns'
            for name, count in [('alpha', 5), ('beta', 2)]:
                (data / name).mkdir(parents=True)
                for index in range(count):
                    (data / name / (str(index) + '.json')).write_text('[]')
            environment = os.environ.copy()
            environment.pop('NEURAL_PATTERN_URL', None)
            environment['NEURAL_PATTERN_DIR'] = str(data)
            result = self.command(['--stats'], directory, environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [line.split() for line in result.stdout.splitlines()]
        by_class = {rows[index][0]: rows[index:index + 3] for index in range(0, len(rows), 3)}
        self.assertEqual(by_class['alpha'], [['alpha', '5'], ['train', '4'], ['test', '1']])
        self.assertEqual(by_class['beta'], [['beta', '2'], ['train', '2'], ['test', '0']])

    def test_multiple_model_orchestration_uses_managed_pool(self):
        # This verifies only process-pool lifetime and dispatch, not model training.
        events = []
        class Pool:
            def __init__(self, count): events.append(('workers', count))
            def __enter__(self): events.append('entered'); return self
            def __exit__(self, *args): events.append('exited')
            def map(self, function, counts): events.append((function.__name__, counts))
        with patch('multiprocessing.Pool', Pool), patch.object(
                sys, 'argv', ['signature.py', '--models', '2', '--train', '3']):
            runpy.run_path(str(ROOT / 'signature.py'), run_name='__main__')
        self.assertEqual(events, [('workers', 2), 'entered', ('test_network', [3, 3]), 'exited'])


if __name__ == '__main__':
    unittest.main()
