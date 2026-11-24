"""Create an aggregate classifier config from recognizer config paths."""

import argparse
import json
from pathlib import Path


def create_config_list(manifest_path: Path, output_path: Path) -> None:
    """Load recognizer JSON files listed in a manifest and write one aggregate config."""
    manifest_path = manifest_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with manifest_path.open('r', encoding='utf-8') as manifest_file:
        manifest = json.load(manifest_file)

    if isinstance(manifest, dict):
        recognizer_paths = manifest.get('recognizer_paths')
        run_config = manifest.get('run_config')
    elif isinstance(manifest, list):
        recognizer_paths = manifest
        run_config = None
    else:
        raise ValueError("Manifest must be an object or a list of recognizer paths")

    if not isinstance(recognizer_paths, list) or not recognizer_paths:
        raise ValueError("Manifest must contain a non-empty 'recognizer_paths' list")
    if not all(isinstance(path, str) and path.strip() for path in recognizer_paths):
        raise ValueError("Each recognizer path must be a non-empty string")

    recognizers = []
    for recognizer_path in recognizer_paths:
        path = Path(recognizer_path).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()

        if not path.is_file():
            raise FileNotFoundError(f"Recognizer config file not found: {path}")

        with path.open('r', encoding='utf-8') as recognizer_file:
            recognizer = json.load(recognizer_file)
        if not isinstance(recognizer, dict):
            raise ValueError(f"Recognizer config must be a JSON object: {path}")
        recognizers.append(recognizer)

    config_list = {'recognizers': recognizers}
    if run_config is not None:
        config_list['run_config'] = run_config

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as output_file:
        json.dump(config_list, output_file, indent=2)
        output_file.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an aggregate classifier config from recognizer JSON paths."
    )
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    create_config_list(args.manifest, args.output)
    print(f"Wrote classifier config list to {args.output.expanduser().resolve()}")


if __name__ == '__main__':
    main()