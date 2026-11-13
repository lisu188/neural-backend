"""Offline fixtures exercise the actual loader without TensorFlow or requests."""
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from data.cache import invalidate_cache
from data import pattern


class PatternSourcesTests(unittest.TestCase):
    def tearDown(self):
        invalidate_cache()

    def test_local_default_has_no_network_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'class-a').mkdir()
            (root / 'class-a' / 'one.json').write_text('[{"X": 1}]')
            with patch.dict(os.environ, {'NEURAL_PATTERN_DIR': temporary}, clear=True):
                self.assertEqual(pattern.load_all(), {'class-a': [
                    {'file': 'one.json', 'data': [{'X': 1}]}]})

    def test_missing_or_empty_data_has_a_clear_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'No class directories'):
                pattern.load_from_disk(temporary)

    def test_preserves_class_indices_and_shuffled_sample_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('z-class', 'a-class'):
                (root / name).mkdir()
                for filename in ('one.json', 'two.json'):
                    (root / name / filename).write_text('[]')
            def listing(path):
                return ['z-class', 'a-class'] if Path(path) == root else ['one.json', 'two.json']
            with patch.object(pattern.os, 'listdir', side_effect=listing), patch.object(
                    pattern, 'shuffled', side_effect=lambda values: reversed(values)) as shuffle:
                result = pattern.load_from_disk(temporary)
            self.assertEqual(list(result), ['z-class', 'a-class'])
            self.assertEqual([item['file'] for item in result['z-class']], ['two.json', 'one.json'])
            self.assertEqual(shuffle.call_count, 2)

    def test_remote_url_is_explicit_and_https(self):
        for value in ('', 'http://example.test/data', 'https://user:pass@example.test/data'):
            with patch.dict(os.environ, {'NEURAL_PATTERN_URL': value}, clear=True):
                with self.assertRaises(ValueError):
                    pattern.remote_url()

    def test_remote_archive_rejects_traversal(self):
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, 'w') as archive:
            archive.writestr('../outside.json', '[]')
        class Response:
            status_code = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_content(self, size): yield payload.getvalue()
        requests = types.ModuleType('requests')
        requests.get = lambda *args, **kwargs: Response()
        with patch.dict('sys.modules', {'requests': requests}), patch.dict(
                os.environ, {'NEURAL_PATTERN_URL': 'https://example.test/patterns'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Unsafe pattern archive'):
                pattern.load_all()


if __name__ == '__main__':
    unittest.main()
