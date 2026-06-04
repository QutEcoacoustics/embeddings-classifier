#!/usr/bin/env python3
"""
Parquet file processor that applies classification model to data.

Processes single files or directories of parquet files, applies a linear
classifier (beta weights + bias), and saves results maintaining folder structure.
Uses pure PyArrow for maximum performance and minimal dependencies.
"""

import argparse
import concurrent.futures
from collections import Counter, deque
import json
import base64
import binascii
from importlib.metadata import PackageNotFoundError, version as get_distribution_version
import os
import sys
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
import io
import re
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pcsv
from urllib.parse import urlparse, urlunparse, ParseResult, parse_qs, urlencode
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

SUPPORTED_EXTENSIONS = ['.parquet', '.csv']
DEFAULT_OUTPUT_EXTENSION = '.csv'
ENV_INPUT_PATH = 'EMBEDDINGS_CLASSIFIER_INPUT'
ENV_OUTPUT_PATH = 'EMBEDDINGS_CLASSIFIER_OUTPUT'
ENV_CONFIG_PATH = 'EMBEDDINGS_CLASSIFIER_CONFIG'

# Keep one HTTP session per worker thread so repeated URL fetches can reuse
# established connections (TCP/TLS keep-alive) and reduce per-item latency.
_thread_local_http = threading.local()


def _get_http_session() -> requests.Session:
    session = getattr(_thread_local_http, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=0)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _thread_local_http.session = session
    return session

@dataclass
class ClassifierItem:
    """Dataclass to hold the result, config, outputpath of a classifier run on a single item."""

    # the normalized config for one single classifier
    config: Dict

    # the output path template (including the <classifier_name> placeholder)
    output_path_template: Optional[Path] = None

    # did the classification run successfully
    success: bool = False

    # the resolved output path for this item 
    output_path: Optional[Path] = None

    # will be populated with the actual classification results table
    result_table: Optional[pa.Table] = None

    # any message to be returned, such as "skipped because output file already exists"
    message: str = ""

    # any error message if success is False
    error: str = ""



    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        # set the output path for this item, based on the output_path_template and the classifier name in the config.
        self.construct_output_path()

        if self.output_path is not None and self.config.get('skip_existing', True) and self.output_path.exists():
            self.success = True
            self.message = f"Output file {self.output_path} already exists, skipping processing."

    def construct_output_path(self):
        """
        replaces the folder "classifier_name" in the output path template with the actual classifier name.
        """

        if self.output_path_template is None:
            self.output_path = None
            return
        
        output_path_template = self.output_path_template
        classifier_name = self.config['classifier_name']

        # use absolute paths for easier debugging
        if not output_path_template.is_absolute():
            output_path_template = Path.cwd() / output_path_template

        # sanitize the classifier name to be a valid folder name
        # replace spaces with underscore and remove everything except [A-Za-z0-9_-]
        classifier_name = re.sub(r'\s+', '_', classifier_name)
        classifier_name = re.sub(r'[^A-Za-z0-9_-]', '', classifier_name)

        # replace the last part of the path with the classifier name
        self.output_path = Path(str(output_path_template).replace('<classifier_name>', classifier_name))
        



def check_output_clashes(resolved_paths):
    """
    For an item's resolved output paths (one per classifier), checks that there were no clashes
    due to duplicate sanitized classifier names. Prevents overwriting results files by multiple classifiers.
    """

    non_none_paths = [path for path in resolved_paths if path is not None]
    duplicate_paths = sorted(path for path, count in Counter(non_none_paths).items() if count > 1)
    if duplicate_paths:
        duplicate_paths_str = ', '.join(str(path) for path in duplicate_paths)
        raise ValueError(
            "Multiple classifiers resolve to the same output path(s) after classifier name sanitization: "
            f"{duplicate_paths_str}."
        )


def resolve_classifier_name(config: Dict[str, Any], index: int) -> str:
    """Resolve classifier name from config with deterministic fallbacks."""
    configured_name = config.get('classifier_name') or config.get('name')
    if isinstance(configured_name, str) and configured_name.strip():
        return configured_name.strip()

    classes = config.get('classifier', {}).get('classes', [])
    if isinstance(classes, list) and len(classes) == 1:
        class_name = re.sub(r'\s+', '_', str(classes[0]).strip().lower())
        return class_name if class_name else f"classifier_{index}"

    if isinstance(classes, list) and len(classes) > 1:
        return f"classifier_{index}_{len(classes)}class"

    return f"classifier_{index}"


@dataclass
class ClassifierConfig:
    """Normalized classifier configuration container."""
    configs: List[Dict[str, Any]]

    @staticmethod
    def _normalize_configs(configs: Any) -> List[Dict[str, Any]]:
        if not isinstance(configs, list):
            configs = [configs]

        normalized_configs = []

        for i, config in enumerate(configs):
            if not isinstance(config, dict):
                raise ValueError(f"Config at index {i} must be an object")

            # Validate required fields
            if 'classifier' not in config:
                # Assume the entire JSON is the classifier and use defaults.
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

            normalized: Dict[str, Any] = dict(config)
            normalized['classifier'] = dict(classifier)

            # defaults
            if 'threshold' not in normalized:
                normalized['threshold'] = 0.0

            if 'save_empty' not in normalized:
                normalized['save_empty'] = True

            if 'skip_existing' not in normalized:
                normalized['skip_existing'] = True

            # Canonical classifier name is resolved once during normalization.
            normalized['classifier_name'] = resolve_classifier_name(normalized, i)

            normalized_configs.append(normalized)

        return normalized_configs

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ClassifierConfig":
        return cls.from_any(config)

    @classmethod
    def from_json(cls, config_path: Union[str, Path]) -> "ClassifierConfig":
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return cls.from_any(raw)

    @classmethod
    def from_any(
        cls,
        config_data: Union["ClassifierConfig", Dict[str, Any], List[Dict[str, Any]], str, Path],
    ) -> "ClassifierConfig":
        if isinstance(config_data, cls):
            return config_data

        if isinstance(config_data, (str, Path)):
            return cls.from_json(config_data)

        if not isinstance(config_data, (dict, list)):
            raise TypeError(f"Unsupported config type: {type(config_data)}")

        normalized_configs = cls._normalize_configs(config_data)
        for config in normalized_configs:
            classifier_config = config['classifier']
            beta, beta_bias = deserialize_classifier_params(classifier_config)
            classifier_config['beta'] = beta
            classifier_config['beta_bias'] = beta_bias

            logging.info("Classes found in config: %s", classifier_config['classes'])
            logging.info("Beta shape: %s, Beta bias shape: %s", beta.shape, beta_bias.shape)

        return cls(configs=normalized_configs)

    def as_list(self) -> List[Dict[str, Any]]:
        return self.configs








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
    subparsers.add_parser('version', help="Show the application version")
    
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
    logging.debug("Environment QSP: %s", env_qsp)
    if env_qsp:
        query_params = parse_qs(parsed_url.query)
        new_query_params = parse_qs(env_qsp)
        # merge dicts
        query_params.update(new_query_params)
        new_query_string = urlencode(query_params, doseq=True)
        parsed_url = parsed_url._replace(query=new_query_string)

    logging.info("Parsed URL: %s", urlunparse(parsed_url))

    return parsed_url



def load_config(config_path: str) -> List[Dict[str, Any]]:
    """Load and validate configuration file."""
    return ClassifierConfig.from_json(config_path).as_list()


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
    
    except (ValueError, TypeError, KeyError, binascii.Error) as e:
        logging.error("Error deserializing classifier parameters: %s", e)
        raise ValueError(f"Error deserializing classifier parameters: {e}") from e



def process_table(
    table: pa.Table, 
    item: ClassifierItem,
    # output_path: Optional[Union[str, Path]], 
    # beta: np.ndarray, 
    # beta_bias: np.ndarray,
    # classes: list,
    # threshold: Union[float, list, dict] = 0.0,
    # save_empty: bool = True,

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

    output_path = Path(item.output_path) if item.output_path is not None else None
    metadata_columns = ['source', 'channel', 'offset']

    # extract classifier parameters from the config for this item for convenience. 
    beta = item.config['classifier']['beta']
    beta_bias = item.config['classifier']['beta_bias']
    classes = item.config['classifier']['classes']
    threshold = item.config['threshold']
    save_empty = item.config['save_empty']

    try:
        
        logging.info("Processing table: %s rows, %s columns", table.num_rows, table.num_columns)

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
            item.success = True
            item.message = "No scores passed the threshold. No output file was created."
            return item
        
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
        item.result_table = result_table
        
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.suffix == '.csv':
                # Save as CSV if the output path ends with .csv
                # result_table.to_pandas().to_csv(output_path, index=False)
                # print(f"Saved results to {output_path}")
                pcsv.write_csv(result_table, output_path,
                               write_options=pcsv.WriteOptions(include_header=True))
            else:
                pq.write_table(result_table, output_path)
            logging.info("Saved results to %s", output_path)
        else:
            logging.info("No output path provided; returning results in memory only.")
        
        item.success = True

    except (ValueError, TypeError, KeyError, OSError, pa.ArrowInvalid, pa.ArrowTypeError) as e:
        logging.error("Error processing table: %s", e)
        item.success = False
        item.error = str(e)





def get_table_from_path(input_path: Union[Path, ParseResult]) -> Tuple[Optional[pa.Table], Optional[str]]:
    """Load a parquet table from a local path or URL.

    Returns a `(table, error)` tuple where `table` is populated on success and
    `error` is `None`; on failure, `table` is `None` and `error` contains a
    descriptive message.
    """

    if isinstance(input_path, Path):
        try:
            table = pq.read_table(input_path)
        except (OSError, pa.ArrowInvalid, pa.ArrowTypeError) as e:
            logging.error("Error reading %s: %s", input_path, e)
            return None, str(e)

    elif isinstance(input_path, ParseResult):
        url = urlunparse(input_path)
        try:
            logging.info("Reading from URL: %s", url)
            # Use explicit connect/read timeouts so workers do not hang indefinitely.
            response = _get_http_session().get(url, timeout=(5, 30))
            response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)
            buffer = io.BytesIO(response.content)
            table = pq.read_table(buffer)

        except requests.exceptions.RequestException as e:
            # Catch network-related errors from the requests library
            logging.error("Error downloading %s: %s", url, e)
            return None, f"Error downloading {url}: {e}"
        except (OSError, pa.ArrowInvalid, pa.ArrowTypeError, ValueError, TypeError) as e:
            # Catch other errors, like pyarrow failing to parse the file
            logging.error("Error processing data from %s: %s", url, e)
            return None, f"Error processing data from {url}: {e}"
    else:
        logging.error("Error: input_path must be a Path or ParseResult, got %s", type(input_path))
        return None, f"Error: input_path must be a Path or ParseResult, got {type(input_path)}"
    
    return table, None


def init_items(
    configs: ClassifierConfig,
    output_path_template: Optional[Path],
) -> List[ClassifierItem]:
    """
    Initializes a list of classifier items (one per classifier) for a single file
    - Checks for output file clashes due to sanitized classifier names
    - Checks if output files already exist and marks those items as successful to skip processing    
    """
    items = [
        ClassifierItem(output_path_template=output_path_template, config=config)
        for config in configs.as_list()
    ]

    check_output_clashes([item.output_path for item in items])     
    return items


def _process_single_input(
    input: Union[pa.Table, Path, ParseResult],
    output_path_template: Optional[Path],
    configs: ClassifierConfig,
) -> List[ClassifierItem]:
    """Process a preloaded table, filepath or URL for all classifier configs."""
 
    # get a dict for each classifier that includes the output path for that classifier for this file. 
    
    # initialise a list of results (1 for each classifier, that shows success if already exists)
    items = init_items(configs, output_path_template)

    # return early
    if all(item.success for item in items):
        logging.info("All output files for %s already exist, skipping processing.", output_path_template)
        return items
    
    # load the table if we were given a path or ParseResult instead of a preloaded table
    if not isinstance(input, pa.Table):
        source = str(input)
        table, error = get_table_from_path(input)
        if table is None:
            logging.error("Error reading table from %s: %s", input, error)
            # for any item that doesn't have results already, set the error message
            for item in items:
                if not item.success: 
                    item.error = str(error)
            return items
    else:
        source = "in_mem_table"
        table = input

    for item in items:

        logging.info("Processing: %s -> %s", source, item.output_path)

        if item.success:
            # already exists, skipping
            continue
      
        logging.info("Processing file: %s for classifier %s", source, item.config['classifier_name'])
        # updates item results in place
        process_table(
            table, 
            item=item
        )

    return items


def classify_table(
    table: pa.Table,
    config: Union[ClassifierConfig, Dict[str, Any], List[Dict[str, Any]], str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> List[ClassifierItem]:
    """Classify an in-memory Arrow table directly."""
    if not isinstance(table, pa.Table):
        raise TypeError(f"Expected pyarrow.Table, got {type(table)}")

    configs = ClassifierConfig.from_any(config)
    output_path_template = None
    if output_path is not None:
        output_path_template = get_full_output_path_templates(
            [Path(output_path)],
            Path('.'),
            [Path('in_memory_table.parquet')],
        )[0]
    return _process_single_input(table, output_path_template=output_path_template, configs=configs)


def classify_dataframe(
    dataframe: Any,
    config: Union[ClassifierConfig, Dict[str, Any], List[Dict[str, Any]], str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> List[ClassifierItem]:
    """Classify a pandas DataFrame by converting it to an Arrow table first."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for classify_dataframe") from e

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(dataframe)}")

    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    return classify_table(table, config, output_path=output_path)

    


def get_parquet_files(directory: Union[Path, str]) -> list:
    """Get all parquet files in directory recursively."""
    files = list(Path(directory).rglob('*.parquet'))
    logging.info("Found %s parquet files", len(files))
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
        with open(input_path, 'r', encoding='utf-8') as f:
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
    
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
        logging.error("Error reading JSON input file: %s", e)
        raise ValueError(f"Error reading JSON input file: {e}") from e



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


def get_full_output_path_templates(
    output_paths: List[Path],
    output_parent: Path,
    input_paths: List[Union[Path, ParseResult]],
) -> List[Path]:
    """Build one output-path template per input, including classifier token placement.

    Token-placement rules:
    - Relative output path: inject <classifier_name> at the left side (near the base)
        unless the relative output path or output_parent already includes the token.
        Example: outputs/result.csv -> <classifier_name>/outputs/result.csv.
    - Absolute output path: inject <classifier_name> at the right side (near the leaf)
        unless the absolute output path already includes the token.
        Example: /root/outputs/result.csv -> /root/outputs/<classifier_name>/result.csv.

    Rationale:
    For absolute paths there is no reliable way to infer a caller-intended split point,
    so classifier injection is anchored at the leaf side for deterministic behavior.

    We need the output path to be templated with the classifier name, because it's possible
    to have multiple classifiers and each needs to output a separate file.  
    """


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
            if '<classifier_name>' in str(op):
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

    configs = ClassifierConfig.from_any(config_path)

    if input_path.is_file():

        logging.info("Processing file: %s", input_path)

        if input_path.suffix == '.parquet':
            input_paths = [input_path]
            output_paths = [output_path]

        elif input_path.suffix == '.json':
            logging.info("Reading input from JSON file: %s", input_path)
            input_paths, output_paths = read_json_input_file(input_path)

        else:
            logging.error("Error: Input file '%s' must be a .parquet or .json file", input_path)
            raise ValueError(f"Input file '{input_path}' must be a .parquet or .json file")

    elif input_path.is_dir():
        input_paths = get_parquet_files(input_path)
        if not input_paths:
            raise RuntimeError(f"No parquet files found in input directory: {input_path}")
        output_paths = [file.relative_to(input_path).with_suffix('.csv') for file in input_paths]

        
    else:
        logging.error("Error: Input path '%s' does not exist", input_path)
        raise FileNotFoundError(f"Input path '{input_path}' does not exist")
               
    success_count = 0
    total_files = len(input_paths)
    completed_count = 0
    start_time = time.monotonic()
    recent_completion_times = deque(maxlen=50)

    def log_progress() -> None:
        if completed_count == 0:
            return
        if completed_count % 20 != 0 and completed_count != total_files:
            return

        # Use a rolling window to avoid misleading rates after restarts with many fast skips.
        if len(recent_completion_times) >= 2:
            window_elapsed_s = recent_completion_times[-1] - recent_completion_times[0]
            window_completed = len(recent_completion_times) - 1
            rate = window_completed / window_elapsed_s if window_elapsed_s > 0 else 0.0
        else:
            elapsed_s = time.monotonic() - start_time
            rate = completed_count / elapsed_s if elapsed_s > 0 else 0.0
        remaining = total_files - completed_count
        eta_min = (remaining / rate / 60.0) if rate > 0 else float('inf')

        if rate > 0:
            logging.info(
                "Progress: %s/%s items processed at %.3f items/s, %.2f mins remaining.",
                completed_count,
                total_files,
                rate,
                eta_min,
            )
        else:
            logging.info(
                "Progress: %s/%s items processed at %.3f items/s, ETA unavailable.",
                completed_count,
                total_files,
                rate,
            )
    
    full_output_path_templates = get_full_output_path_templates(output_paths, Path(output_path), input_paths)

    worker_count = max(1, int(workers))

    if worker_count == 1:
        for input_path, output_path_template in zip(input_paths, full_output_path_templates):
            try:
                file_results = _process_single_input(
                    input_path, output_path_template=output_path_template, configs=configs
                )
            except Exception as exc:
                logging.error("--> FAILED: %s: %s", input_path, exc, exc_info=True)
            else:
                if any([r.success is False for r in file_results]):
                    logging.warning("--> FAILED: %s", input_path)
                else:
                    success_count += 1

            completed_count += 1
            recent_completion_times.append(time.monotonic())
            log_progress()
    else:
        logging.info("Processing %s files with %s workers", total_files, worker_count)
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_input = {
                executor.submit(_process_single_input, input_path, output_path_template=output_path_template, configs=configs): input_path
                for input_path, output_path_template in zip(input_paths, full_output_path_templates)
            }

            for future in concurrent.futures.as_completed(future_to_input):
                current_input = future_to_input[future]
                try:
                    file_results = future.result()
                except Exception as exc:
                    logging.error("--> FAILED: %s: %s", current_input, exc, exc_info=True)
                else:
                    if any([r.success is False for r in file_results]):
                        logging.warning("--> FAILED: %s", current_input)
                    else:
                        success_count += 1

                completed_count += 1
                recent_completion_times.append(time.monotonic())
                log_progress()

    logging.info("\n--- PROCESSING SUMMARY ---")
    logging.info("Successfully processed %s/%s files.", success_count, total_files)

    if success_count < total_files:
        failure_count = total_files - success_count
        logging.warning("\nEncountered %s error(s) during processing. Exiting with error code 1.", failure_count)
        raise RuntimeError(f"Encountered {failure_count} error(s) during processing")

    logging.info("\nAll files processed successfully.")
            



def show_version():
    """Print application version from installed package metadata."""
    logging.info("Running 'version' command...")

    try:
        print(f"Version: {get_distribution_version('embeddings-classifier')}")
        return
    except PackageNotFoundError as e:
        raise RuntimeError(
            "Version metadata not found. Install the package to use the 'version' command."
        ) from e


def get_paths(args):
    """
    Resolve CLI input/output/config paths from args, then environment variables.

    Precedence for each path is:
    1) explicit CLI argument
    2) environment variable fallback

    Raises:
        ValueError: if any required path cannot be resolved.
    """

    input_path = args.input if args.input else os.environ.get(ENV_INPUT_PATH)
    output_path = args.output if args.output else os.environ.get(ENV_OUTPUT_PATH)
    config_path = args.config if args.config else os.environ.get(ENV_CONFIG_PATH)

    missing = []
    if input_path is None:
        missing.append(f"--input or ${ENV_INPUT_PATH}")
    if output_path is None:
        missing.append(f"--output or ${ENV_OUTPUT_PATH}")
    if config_path is None:
        missing.append(f"--config or ${ENV_CONFIG_PATH}")

    if missing:
        raise ValueError(
            "Missing required classify paths. Provide " + ", ".join(missing) + "."
        )

    assert input_path is not None
    assert output_path is not None
    assert config_path is not None

    return Path(input_path), Path(output_path), Path(config_path)


def main():
    """Parses arguments and calls the appropriate function."""
    args = parse_arguments()

    try:
        if args.command == 'classify':
            input_path, output_path, config_path = get_paths(args)
            classify(input_path, output_path, config_path, args.workers)
        elif args.command == 'version':
            show_version()
    except Exception as e:
        logging.error("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()