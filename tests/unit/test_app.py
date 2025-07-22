import sys
from unittest.mock import patch
from pathlib import Path
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import TestHelpers

import app 


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
    with patch('app.classify') as mock_classify:
        app.main()
        
        # Assert that our mock function was called exactly once
        mock_classify.assert_called_once()
        
        # Assert it was called with the correct Path objects
        mock_classify.assert_called_once_with(
            Path('in.pq'), Path('out'), Path('conf.json')
        )

def test_main_calls_show_version(monkeypatch):
    """
    Tests that `main` calls the `show_version` function
    when 'version' subcommand is used.
    """
    monkeypatch.setattr(sys, 'argv', ['app.py', 'version'])
    
    with patch('app.show_version') as mock_show_version:
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

