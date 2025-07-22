import base64
import json
from pathlib import Path
import pytest
from typing import Union
import shutil
from functools import reduce
import shlex
import subprocess


class TestHelpers:
    """Helper functions for test setup and teardown."""
    
    @staticmethod
    def get_test_dirs(folder = None):
        """Get the test directory paths."""
        test_root = Path(__file__).parent
        folders = {
            'data_parquet': test_root / 'test_data' / 'parquet',
            'data_config': test_root / 'test_data' / 'config',
            'data_output': test_root / 'test_data' / 'output',
            'workspace_input': test_root / 'workspace' / 'input',
            'workspace_config': test_root / 'workspace' / 'config',
            'workspace_output': test_root / 'workspace' / 'output'
        }
    
        if folder:
            return folders[folder]
        else:
            return folders
    

    @staticmethod
    def clean_mounted_directory(directory_path: Path):
        """Remove all files except .gitkeep from a directory."""
        if not directory_path.exists():
            directory_path.mkdir(parents=True, exist_ok=True)
            return
        
        for item in directory_path.iterdir():
            if item.name != '.gitkeep':
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
    


    @staticmethod
    def copy_input(source_file, dest_file = None):
        """Copy files from source to destination directory."""

        TestHelpers.copy_test_file(TestHelpers.get_test_dirs('data_parquet'), 
                            TestHelpers.get_test_dirs('workspace_input'), 
                            source_file, dest_file)

    @staticmethod
    def copy_config(source_file, dest_file = None):
        """Copy files from source to destination directory."""

        TestHelpers.copy_test_file(TestHelpers.get_test_dirs('data_config'), 
                                   TestHelpers.get_test_dirs('workspace_config'), 
                                   source_file, dest_file)


    @staticmethod
    def copy_test_file(source_parent, dest_parent, source_file, dest_file = None):
        """
        Copy a file from source to destination, ensuring parent directories exist.
        If dest_file is None, will use the name of the source file.
        """
        
        if dest_file is None:
            dest_file = Path(source_file).name
        dest_file = Path(dest_parent) / dest_file
        source_file = Path(source_parent) / source_file
        # dest_file might include a subdirectory structure, ensure parent exists
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, dest_file)


    @staticmethod
    def sys_command(command: Union[str, list], cwd: Union[str, Path] = None):
        """
        Execute a system command and return the output.
        If command is a string, it will be split into a list (using shlex.split for safety).
        Generates a bash-friendly string for easy copy-pasting.
        """
        # Ensure command is a list of strings
        if isinstance(command, str):
            command_list = shlex.split(command)
        else:
            command_list = [str(arg) for arg in command]
        command_for_bash = ' '.join([shlex.quote(arg) for arg in command_list])
        print(f"Executing command: {command_for_bash} in {cwd if cwd else 'current directory'}")
 
        try:
            result = subprocess.run(command_list, capture_output=True, text=True, cwd=cwd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command failed with return code {e.returncode}:\n{e.stderr.strip()}")
        except FileNotFoundError:
            raise RuntimeError(f"Command not found: '{command_list[0]}'. Please ensure it's in your PATH.")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while executing command: {e}")

        return result.stdout.strip(), result.stderr.strip()

