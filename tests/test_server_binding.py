"""Exercise server startup without loading historical TensorFlow dependencies."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class FakeFlask:
    def __init__(self, name):
        self.routes = []

    def route(self, *args, **kwargs):
        self.routes.append((args, kwargs))
        return lambda function: function

    def run(self, **kwargs):
        self.run_options = kwargs


def module_stub(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


class ServerBindingTests(unittest.TestCase):
    def start(self, environment):
        app = FakeFlask('neural')
        stubs = {
            'tensorflow': module_stub('tensorflow'),
            'flask': module_stub('flask', Flask=lambda name: app, jsonify=None, request=None),
            'data': module_stub('data'),
            'data.cache': module_stub('data.cache', invalidate_cache=None),
            'data.config': module_stub('data.config', VECTORS=None),
            'data.conversion': module_stub('data.conversion', convert_pattern=None),
            'data.operations': module_stub('data.operations', average_pattern=None, rms=None, avg=None),
            'data.pattern': module_stub('data.pattern', load_all=None),
            'engine': module_stub('engine'),
            'engine.runner': module_stub('engine.runner', build_network=None),
        }
        source = Path(__file__).resolve().parents[1] / 'rest/server.py'
        with patch.dict(sys.modules, stubs), patch.dict(os.environ, environment, clear=True):
            spec = importlib.util.spec_from_file_location('server_binding_under_test', source)
            server = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(server)
            # Avoid constructing a model: this test exercises only the actual
            # HTTP startup method and its configuration boundary.
            engine = server.NeuralRestEngine.__new__(server.NeuralRestEngine)
            engine.lock = object()
            engine.run_server()
        self.assertIn((('/train/<int:steps>',), {'methods': ['POST'], 'endpoint': 'train'}), app.routes)
        self.assertIn((('/reload',), {'methods': ['POST'], 'endpoint': 'reload'}), app.routes)
        return app.run_options

    def test_default_listener_is_loopback(self):
        self.assertEqual(self.start({}), {'host': '127.0.0.1', 'port': 5000})

    def test_port_alone_does_not_expose_listener(self):
        self.assertEqual(self.start({'PORT': '8123'}), {'host': '127.0.0.1', 'port': 8123})

    def test_remote_binding_requires_explicit_configuration(self):
        self.assertEqual(self.start({'NEURAL_HOST': '0.0.0.0', 'PORT': '8123'}),
                         {'host': '0.0.0.0', 'port': 8123})

    def test_empty_host_keeps_loopback_default(self):
        self.assertEqual(self.start({'NEURAL_HOST': ''})['host'], '127.0.0.1')


if __name__ == '__main__':
    unittest.main()
