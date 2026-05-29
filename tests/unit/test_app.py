import sys
from unittest.mock import patch
from pathlib import Path
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

import embeddings_classifier.app as app


def test_main_calls_classify_with_args(monkeypatch):
    """
    Tests that `main` calls the `classify` function with the correct arguments
    when 'classify' subcommand is used.
    """
    # Use monkeypatch to simulate command-line arguments
    monkeypatch.setattr(
        sys, 'argv', 
        ['app.py', 'classify', '--input', 'in.pq', '--output', 'out', '--config', 'conf.json']
    )
    
    # Patch the actual logic function so we only test the CLI wiring
    with patch('embeddings_classifier.app.classify') as mock_classify:
        app.main()
        
        # Assert that our mock function was called exactly once
        mock_classify.assert_called_once()
        
        # Assert it was called with the correct Path objects
        mock_classify.assert_called_once_with(
            Path('in.pq'), Path('out'), Path('conf.json'), 1
        )

def test_main_calls_show_version(monkeypatch):
    """
    Tests that `main` calls the `show_version` function
    when 'version' subcommand is used.
    """
    monkeypatch.setattr(sys, 'argv', ['app.py', 'version'])
    
    with patch('embeddings_classifier.app.show_version') as mock_show_version:
        app.main()
        mock_show_version.assert_called_once()

def test_main_exits_on_invalid_subcommand(monkeypatch):
    """
    Tests that argparse exits when an unknown subcommand is provided.
    """
    monkeypatch.setattr(sys, 'argv', ['app.py', 'invalid-command'])
    
    with pytest.raises(SystemExit) as e:
        app.main()
    
    # Argparse exits with code 2 for bad arguments
    assert e.value.code == 2

def test_main_exits_on_unrecognized_argument(monkeypatch):
    """
    Tests that argparse exits when an unrecognized argument is provided.
    """
    monkeypatch.setattr(sys, 'argv', ['app.py', 'classify', '--this-is-not-an-argument'])
    
    with pytest.raises(SystemExit):
        app.main()


def test_get_paths_uses_env_fallbacks(monkeypatch):
    """get_paths should use environment defaults when args are omitted."""
    monkeypatch.setenv('EMBEDDINGS_CLASSIFIER_INPUT', '/tmp/in')
    monkeypatch.setenv('EMBEDDINGS_CLASSIFIER_OUTPUT', '/tmp/out')
    monkeypatch.setenv('EMBEDDINGS_CLASSIFIER_CONFIG', '/tmp/conf.json')

    args = type('Args', (), {'input': None, 'output': None, 'config': None})()

    input_path, output_path, config_path = app.get_paths(args)

    assert input_path == Path('/tmp/in')
    assert output_path == Path('/tmp/out')
    assert config_path == Path('/tmp/conf.json')


def test_get_paths_raises_when_missing(monkeypatch):
    """get_paths should fail if neither args nor env vars provide required paths."""
    monkeypatch.delenv('EMBEDDINGS_CLASSIFIER_INPUT', raising=False)
    monkeypatch.delenv('EMBEDDINGS_CLASSIFIER_OUTPUT', raising=False)
    monkeypatch.delenv('EMBEDDINGS_CLASSIFIER_CONFIG', raising=False)

    args = type('Args', (), {'input': None, 'output': None, 'config': None})()

    with pytest.raises(ValueError):
        app.get_paths(args)

