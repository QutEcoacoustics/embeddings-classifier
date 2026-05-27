#!/usr/bin/env python3
"""
Parquet file processor that applies classification model to data.

Processes single files or directories of parquet files, applies a linear
classifier (beta weights + bias), and saves results maintaining folder structure.
Uses pure PyArrow for maximum performance and minimal dependencies.
"""

import argparse
import concurrent.futures
import json
import base64
from importlib.metadata import PackageNotFoundError, version as get_distribution_version
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple
import requests
import io
import re
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pcsv
import pyarrow.compute as pc
import pyarrow.fs as fs
from urllib.parse import urlparse, urlunparse, ParseResult, parse_qs, urlencode
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

SUPPORTED_EXTENSIONS = ['.parquet', '.csv']
DEFAULT_OUTPUT_EXTENSION = '.csv'

@dataclass
class ClassifierResult:
    """Dataclass to hold the result of a classifier run."""
    success: bool
    output_path: Optional[Path] = None
    message: str = ""
    error: str = ""






def parse_arguments():
    """Parse command line arguments with subcommands."""
    parser = argparse.ArgumentParser(
        description="Process parquet files or show version information."
    )
    # This ensures a subcommand is required
    subparsers = parser.add_subparsers(dest='command', required=True, help="Available commands")

    # --- 'classify' subcommand ---
    parser_classify = subparsers.add_parser('classify', help="Run classification on parquet files")
    parser_classify.add_argument(
        "--input", "-i",
        required=False,
        help="Input parquet file or directory"
    )
    parser_classify.add_argument(
        "--output", "-o", 
        required=False,
        help="Output file or directory for results"
    )
    parser_classify.add_argument(
        "--config", "-c",
        required=False,
        help="JSON configuration file with classifier parameters"
    )
    parser_classify.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of files to process in parallel (default: 1)"
    )

    # --- 'version' subcommand ---
    parser_version = subparsers.add_parser('version', help="Show the application version")
    
    return parser.parse_args()



def get_parsed_url(url: str) -> ParseResult:
    """Parse a URL and return a ParseResult object.
       Validates the url by checking if it has a netloc (domain).
       adds any query string parameters provded as env variables to the url.
    """
    
    parsed_url = urlparse(url)
    if not parsed_url.netloc:
        raise ValueError(f"Invalid URL: {url} could not determine domain (netloc).")
    
    # add any env qsps to the url
    env_qsp = os.environ.get('QSP', None)
    logging.debug(f"Environment QSP: {env_qsp}")
    if env_qsp:
        query_params = parse_qs(parsed_url.query)
        new_query_params = parse_qs(env_qsp)
        # merge dicts
        query_params.update(new_query_params)
        new_query_string = urlencode(query_params, doseq=True)
        parsed_url = parsed_url._replace(query=new_query_string)

    logging.info(f"Parsed URL: {urlunparse(parsed_url)}")

    return parsed_url



def load_config(config_path: str) -> List[Dict[str, Any]]:
    """Load and validate configuration file."""

    try:

        with open(config_path, 'r') as f:
            configs = json.load(f)

        if not isinstance(configs, list):
            configs = [configs]

        normalized_configs = []

        for i, config in enumerate(configs):

            # Validate required fields
            if 'classifier' not in config:

                # assume that the entire json is the classifier, and use defaults for # threshold and save_empty
                if 'classes' not in config:
                    raise ValueError("Config must contain 'classifier' section")
                
                config = {
                    'classifier': config
                }
            
            classifier = config['classifier']
            required_fields = ['classes', 'beta', 'beta_bias']
            for field in required_fields:
                if field not in classifier:
                    raise ValueError(f"Classifier must contain '{field}' field")
                
            if 'classifier_name' not in config:
                config['classifier_name'] = f"classifier_{i}"
                
            # defaults
            if 'threshold' not in config:
                config['threshold'] = 0.0

            if 'save_empty' not in config:
                config['save_empty'] = True

            if 'skip_existing' not in config:
                config['skip_existing'] = True

            normalized_configs.append(config)

        return normalized_configs

    except FileNotFoundError:
        logging.error(f"Error: Config file '{config_path}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        sys.exit(1)


def deserialize_classifier_params(classifier_config: Dict[str, Any]) -> tuple:
    """Deserialize beta weights and bias from base64 encoded strings."""
    
    def do_decode(x: str) -> np.ndarray:
        flat = np.frombuffer(base64.b64decode(x.encode('ascii')), dtype=np.float32)
        return flat
        
    try:
        beta_flat = do_decode(classifier_config['beta'])
        num_rows = len(beta_flat) // len(classifier_config['classes'])
        num_cols = len(classifier_config['classes'])
        beta = beta_flat.reshape(num_rows, num_cols)
        beta_bias = do_decode(classifier_config['beta_bias'])
        return beta, beta_bias
    
    except Exception as e:
        logging.error(f"Error deserializing classifier parameters: {e}")
        sys.exit(1)



def process_table(
    table: pa.Table, 
    output_path: Union[str, Path], 
    beta: np.ndarray, 
    beta_bias: np.ndarray,
    classes: list,  # New parameter for the list of class names
    threshold: Union[float, list, dict] = 0.0,
    save_empty: bool = True
):
    """
    Process a single parquet file
    @param input_path: Path to the input parquet file
    @param output_path: Path to save the output csv file
    @param beta: Classifier weights as a numpy array
    @param beta_bias: Classifier bias as a numpy array
    @param classes: List of class names for classification
    @param threshold: Classification threshold (float for all classes, or list or dict of floats for threshold per class) (default 0.0)
    @param save_empty: Whether to save empty results (default True)
    @return: True if processing was successful, False otherwise
    """

    output_path = Path(output_path)
    metadata_columns = ['source', 'channel', 'offset']

    result = ClassifierResult(
        success=False,
        output_path=output_path
    )

    try:

        
        logging.info(f"Processing table: {table.num_rows} rows, {table.num_columns} columns")

        num_feature_cols = len(table.column_names) - len(metadata_columns)

        if num_feature_cols != beta.shape[0]:
            raise ValueError(f"File has only {num_feature_cols} feature columns, classifier expects {beta.shape[0]} features")
        
        feature_column_names = table.column_names[len(metadata_columns):len(metadata_columns)+num_feature_cols]
        feature_arrays = [table.column(col_name).to_numpy(zero_copy_only=False).astype(np.float32) for col_name in feature_column_names]
        features = np.column_stack(feature_arrays)
        
        #TODO: figure out if beta_bias shape should be like (3,) (one dimensional) or (3,1) (two dimensional)
        # 1d is what we were using, but 2d means we can use a general decoder function
        if beta.shape != (num_feature_cols, len(classes)) or beta_bias.shape != (len(classes),):
            raise ValueError("Dimension mismatch between beta/bias and features/classes.")
        
        # produces a (num_rows, num_classes) matrix
        scores = features @ beta + beta_bias

        if isinstance(threshold, float):
            threshold_array = np.full(len(classes), threshold, dtype=np.float32)
        elif isinstance(threshold, list):
            if len(threshold) != len(classes):
                raise ValueError(f"Threshold list length ({len(threshold)}) must match number of classes ({len(classes)})")
            threshold_array = np.array(threshold, dtype=np.float32)
        elif isinstance(threshold, dict):
            # for each class in the threshold dict, use its specified threshold, otherwise use 0.0
            threshold_array = np.array([threshold.get(cls, 0.0) for cls in classes], dtype=np.float32)
        elif threshold is None:
            threshold_array = np.full(len(classes), np.finfo(np.float32).min, dtype=np.float32)
        else:
            raise TypeError("Threshold must be a float or a list of floats")
        
        above_threshold_mask = scores >= threshold_array
        # This gives us two 1D arrays: one for the row index, one for the class index
        passing_row_indices, passing_class_indices = np.where(above_threshold_mask)

        if len(passing_row_indices) == 0 and not save_empty:
            logging.info("No scores passed the threshold. No output file will be created.")
            result.success = True
            result.message = "No scores passed the threshold. No output file was created."
            return result
        
        scores_long = scores[passing_row_indices, passing_class_indices]
        classes_long = np.array(classes)[passing_class_indices]

        # Select the metadata from the original rows that had at least one passing score
        repeated_metadata_arrays = []
        for col_name in metadata_columns:
            original_col_numpy = table.column(col_name).to_numpy(zero_copy_only=False)
            # Use the row indices to select the correct metadata values
            passing_metadata_col = original_col_numpy[passing_row_indices]
            repeated_metadata_arrays.append(pa.array(passing_metadata_col))
            
        # Assemble the final Arrow table from the newly created long-format columns.
        result_columns = repeated_metadata_arrays + [
            pa.array(classes_long),
            pa.array(scores_long, type=pa.float32())
        ]
        result_column_names = metadata_columns + ['label', 'score']
        
        result_table = pa.table(result_columns, names=result_column_names)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == '.csv':
            # Save as CSV if the output path ends with .csv
            # result_table.to_pandas().to_csv(output_path, index=False)
            # print(f"Saved results to {output_path}")
            pcsv.write_csv(result_table, output_path,
                           write_options=pcsv.WriteOptions(include_header=True))
        else:
            pq.write_table(result_table, output_path)
        logging.info(f"Saved results to {output_path}")
        
        result.success = True

        return result

    except Exception as e:
        logging.error(f"Error processing table: {e}")
        result.success = False
        result.error = str(e)
        return result


def construct_output_path(output_path_template: Path, classifier_name):
    """
    replaces the folder "classifier_name" in the output path template with the actual classifier name.
    """
    if not output_path_template.is_absolute():
        output_path_template = Path.cwd() / output_path_template

    # sanitize the classifier name to be a valid folder name
    # replace spaces with underscore and remove everything except [A-Za-z0-9_-]
    classifier_name = re.sub(r'\s+', '_', classifier_name)
    classifier_name = re.sub(r'[^A-Za-z0-9_-]', '', classifier_name)

    # replace the last part of the path with the classifier name
    output_path = Path(str(output_path_template).replace('<classifier_name>', classifier_name))
    
    return output_path


def get_table_from_path(input_path: Union[Path, ParseResult]) -> Tuple[Optional[pa.Table], Optional[str]]:
    """Load a parquet table from a local path or URL.

    Returns a `(table, error)` tuple where `table` is populated on success and
    `error` is `None`; on failure, `table` is `None` and `error` contains a
    descriptive message.
    """

    if isinstance(input_path, Path):
        try:
            table = pq.read_table(input_path)
        except Exception as e:
            logging.error(f"Error reading {input_path}: {e}")
            return None, str(e)

    elif isinstance(input_path, ParseResult):
        url = urlunparse(input_path)
        try:
            logging.info(f"Reading from URL: {url}")
            # Use explicit connect/read timeouts so workers do not hang indefinitely.
            response = requests.get(url, timeout=(5, 30))
            response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)
            buffer = io.BytesIO(response.content)
            table = pq.read_table(buffer)

        except requests.exceptions.RequestException as e:
            # Catch network-related errors from the requests library
            logging.error(f"Error downloading {url}: {e}")
            return None, f"Error downloading {url}: {e}"
        except Exception as e:
            # Catch other errors, like pyarrow failing to parse the file
            logging.error(f"Error processing data from {url}: {e}")
            return None, f"Error processing data from {url}: {e}"
    else:
        logging.error(f"Error: input_path must be a Path or ParseResult, got {type(input_path)}")
        return None, f"Error: input_path must be a Path or ParseResult, got {type(input_path)}"
    
    return table, None



def process_single_file(
    input_path: Union[Path, ParseResult], 
    output_path: Path, 
    configs: List[Dict[str, Any]],
) -> List[ClassifierResult]:
    """
    Process a single parquet file
    @param input_path: Path to or url to the input parquet file
    @param output_path: Path to save the output csv file. This is a template, where 'classifier_name' will be replaced with the classifier name from the config.
    """

    # Build per-file config views so shared config objects are not mutated across threads.
    configs_with_outputs = []
    for config in configs:
        config_for_file = dict(config)
        config_for_file['output_path'] = construct_output_path(output_path, config['classifier_name'])
        configs_with_outputs.append(config_for_file)

    results = [ClassifierResult(success=False, output_path=config['output_path']) for config in configs_with_outputs]

    # return early if everything is done
    if all(co['output_path'].exists() and co['skip_existing'] for co in configs_with_outputs):
        logging.info(f"All output files for {input_path} already exist, skipping processing.")
        for result in results:
            result.success = True
            result.message = f"Output file {result.output_path} already exists, skipping processing."
        return results


    table, error = get_table_from_path(input_path)
    if table is None:
        logging.error(f"Error reading table from {input_path}: {error}")
        for result in results:
            result.success = False
            result.error = str(error)
        return results

    for i, config in enumerate(configs_with_outputs):

        logging.info(f"Processing: {input_path} -> {output_path}")

        if config['skip_existing'] and config['output_path'].exists():
            logging.info(f"Output file {config['output_path']} already exists, skipping processing.")
            results[i] = ClassifierResult(
                success=True,
                output_path=config['output_path'],
                message=f"Output file {config['output_path']} already exists, skipping processing."
            )
            continue
      
        logging.info(f"Processing file: {input_path} for classifier {config['classifier_name']}")
        result = process_table(
            table, config['output_path'], config['classifier']['beta'], config['classifier']['beta_bias'], 
            config['classifier']['classes'], config['threshold'], config['save_empty']
        )
        results[i] = result

    return results

    


def get_parquet_files(directory: Union[Path, str]) -> list:
    """Get all parquet files in directory recursively."""
    files = list(Path(directory).rglob('*.parquet'))
    logging.info(f"Found {len(files)} parquet files")
    return files


def read_json_input_file(input_path: Union[Path, str]) -> tuple:
    """
    Read a JSON file containing URLs and output paths.
    The JSON should be a list of objects with 'url' and 'output_path' keys.

    The json is expected to be in the following format:
    {
        "source": [...],
        "output": [...]
    }

    output is optional. If provided should be parallel to the source list and provide an output for each source. 
    If not provided, the output path will be created based on the source.    
    
    """
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            raise ValueError("Input JSON must be an object in the format {'source': [...], 'output': [...]} (where 'output' is optional)")
        
        input_paths = data.get('source', [])
        output_paths = data.get('output')

        if not input_paths:
            raise ValueError("Input JSON must contain 'source' key with a list of URLs")
        else:
            input_paths = [get_parsed_url(url) for url in input_paths]

        if output_paths is None:
            output_paths = [url_to_local_path(url) for url in input_paths]
    
        if len(input_paths) != len(output_paths):
            raise ValueError(f"Input (len {len(input_paths)}) and output (len {len(output_paths)}) lists must have the same length")
        
        output_paths = [Path(op) for op in output_paths]
                
        return input_paths, output_paths
    
    except Exception as e:
        logging.error(f"Error reading JSON input file: {e}")
        sys.exit(1)



def url_to_local_path(parsed_url: ParseResult, output_extension: str = '.csv') -> Path:
    """Convert a URL to a local file path.
    e.g. 'https://example.com/path/to/resource' -> 'example.com/path/to/resource.csv'
    """
    
    domain_dir = parsed_url.netloc.replace(':', '_')
    path_segments = [segment for segment in parsed_url.path.split('/') if segment]

    if not path_segments:
        # no path segments, use default filename in the domain directory
        local_dir_path = Path(domain_dir)
        filename_base = "index"
    else:
        local_dir_path = Path(domain_dir).joinpath(*path_segments[:-1])
        filename_base = path_segments[-1]

    output_filename = f"{Path(filename_base).stem}{output_extension}"

    full_relative_path = local_dir_path / output_filename
    
    return full_relative_path


def input_path_stem(input_path: Union[Path, ParseResult]) -> str:
    """Return a stable filename stem for local paths and URL parse results."""
    if isinstance(input_path, Path):
        return input_path.stem
    if isinstance(input_path, ParseResult):
        leaf = Path(input_path.path).name
        return Path(leaf).stem if leaf else "index"
    return "input"


def full_output_path_templates(
    output_paths: List[Path],
    output_parent: Path,
    input_paths: List[Union[Path, ParseResult]],
) -> List[Path]:
    # construct an absoulte output path for each input path
    # if the given output path is relative, everything goes under a folder for each classifier
    # if the given output path is absolute, the final leaf will be put under a folder for each classifier
    # unless the output path OR output parent already contains a folder named <classifier_name>


    full_output_paths = []
    for i, op in enumerate(output_paths):
        # if the output path does not have a supported extension, assume it's a directory
        # and give it a filename based on the input path and the default extension
        if op.suffix not in SUPPORTED_EXTENSIONS:
            op = op / (f"{input_path_stem(input_paths[i])}{DEFAULT_OUTPUT_EXTENSION}")
        if not op.is_absolute():
            if '<classifier_name>' in str(op) or '<classifier_name>' in str(output_parent):
                full_output_paths.append(output_parent / op)
            else:
                full_output_paths.append(output_parent / '<classifier_name>' / op)
        else:
            if '<classifier_name>' in str(op) or '<classifier_name>' in str(output_parent):
                full_output_paths.append(op)
            else:
                full_output_paths.append(op.parent / '<classifier_name>' / op.name)

    return full_output_paths



def classify(input_path, output_path, config_path, workers=1):
    """
    Main processing function for the 'classify' command.
    @param input_path: 
      - Path to the input parquet file or directory
      - Path to a json file of urls and output paths
    """

    logging.debug('classify command called')

    configs = load_config(config_path)

    for config in configs:
        classifier_config = config['classifier']
        beta, beta_bias = deserialize_classifier_params(classifier_config)
        classifier_config['beta'] = beta
        classifier_config['beta_bias'] = beta_bias
    
        logging.info(f"Classes found in config: {classifier_config['classes']}")
        logging.info(f"Beta shape: {beta.shape}, Beta bias shape: {beta_bias.shape}")

    if input_path.is_file():

        logging.info(f"Processing file: {input_path}")

        if input_path.suffix == '.parquet':
            input_paths = [input_path]
            output_paths = [output_path]

        elif input_path.suffix == '.json':
            logging.info(f"Reading input from JSON file: {input_path}")
            input_paths, output_paths = read_json_input_file(input_path)

        else:
            logging.error(f"Error: Input file '{input_path}' must be a .parquet or .json file")
            sys.exit(1)

    elif input_path.is_dir():
        input_paths = get_parquet_files(input_path)
        if not input_paths:
            sys.exit(1)
        output_paths = [file.relative_to(input_path).with_suffix('.csv') for file in input_paths]

        
    else:
        logging.error(f"Error: Input path '{input_path}' does not exist")
        sys.exit(1)
               
    success_count = 0
    total_files = len(input_paths)
    
    full_output_paths = full_output_path_templates(output_paths, Path(output_path), input_paths)


    worker_count = max(1, int(workers))

    if worker_count == 1:
        for input_path, output_path in zip(input_paths, full_output_paths):
            file_results = process_single_file(
                input_path, output_path, configs
            )

            if any([r.success is False for r in file_results]):
                logging.warning(f"--> FAILED: {input_path}")
            else:
                success_count += 1
    else:
        logging.info(f"Processing {total_files} files with {worker_count} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_input = {
                executor.submit(process_single_file, input_path, output_path, configs): input_path
                for input_path, output_path in zip(input_paths, full_output_paths)
            }

            for future in concurrent.futures.as_completed(future_to_input):
                current_input = future_to_input[future]
                try:
                    file_results = future.result()
                except Exception as exc:
                    logging.error(f"--> FAILED: {current_input}: {exc}", exc_info=True)
                    continue

                if any([r.success is False for r in file_results]):
                    logging.warning(f"--> FAILED: {current_input}")
                else:
                    success_count += 1

    logging.info("\n--- PROCESSING SUMMARY ---")
    logging.info(f"Successfully processed {success_count}/{total_files} files.")

    if success_count < total_files:
        failure_count = total_files - success_count
        logging.warning(
            f"\nEncountered {failure_count} error(s) during processing. "
            "Exiting with error code 1."
        )
        sys.exit(1)

    logging.info("\nAll files processed successfully.")
            



def show_version():
    """Prints version from package metadata, with file fallbacks for dev/container use."""
    logging.info("Running 'version' command...")
    try:
        print(f"Version: {get_distribution_version('embeddings-classifier')}")
        return
    except PackageNotFoundError:
        pass

    version_file_container = Path('/VERSION')
    version_file_src = Path(__file__).parent.parent / 'VERSION'

    if version_file_container.exists():
        print(f"Version: {version_file_container.read_text().strip()}")
        return
    if version_file_src.exists():
        print(f"Version: {version_file_src.read_text().strip()}")
        return

    print("Error: version metadata is unavailable and VERSION file was not found.", file=sys.stderr)
    sys.exit(1)


def using_container():
    """
    Check if the script is running inside a container by checking if the 
    /VERSION exists and matches the source VERSION file.
    """
    container_version_file = Path('/VERSION')
    source_version_file = Path(__file__).parent.parent / 'VERSION'
    if not container_version_file.exists() or not source_version_file.exists():
        return False
    if container_version_file.read_text().strip() != source_version_file.read_text().strip():
        return False
    return True


def get_paths(args):
    """
    Return default paths for input, output, and config based on whether running in a container.
    """

    default_paths = {
        'input': '/mnt/input',
        'output': '/mnt/output',
        'config': '/mnt/config'
    }

    input_path = args.input if args.input else default_paths['input']
    output_path = args.output if args.output else default_paths['output']
    config_path = args.config if args.config else Path(default_paths['config']) / 'config.json'

    return Path(input_path), Path(output_path), Path(config_path)


def main():
    """Parses arguments and calls the appropriate function."""
    args = parse_arguments()
    
    if args.command == 'classify':
        input_path, output_path, config_path = get_paths(args)
        classify(input_path, output_path, config_path, args.workers)
    elif args.command == 'version':
        show_version()


if __name__ == "__main__":
    main()