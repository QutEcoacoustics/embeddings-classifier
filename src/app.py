#!/usr/bin/env python3
"""
Parquet file processor that applies classification model to data.

Processes single files or directories of parquet files, applies a linear
classifier (beta weights + bias), and saves results maintaining folder structure.
Uses pure PyArrow for maximum performance and minimal dependencies.
"""

import argparse
import json
import base64
import os
import sys
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pcsv
import pyarrow.compute as pc


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

    # --- 'version' subcommand ---
    parser_version = subparsers.add_parser('version', help="Show the application version")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration file."""

    try:

        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        if 'classifier' not in config:

            # assume that the entire json is the classifier, and use defaults for # threshold and save_empty
            if 'classes' not in config:
                raise ValueError("Config must contain 'classifier' section")
            
            config = {
                'classifier': config,
                'threshold': 0.0,
                'save_empty': True
            }
        
        classifier = config['classifier']
        required_fields = ['classes', 'beta', 'beta_bias']
        for field in required_fields:
            if field not in classifier:
                raise ValueError(f"Classifier must contain '{field}' field")
            
        # defaults
        if 'threshold' not in config:
            config['threshold'] = 0.0
        
        return config
    
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
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
        print(f"Error deserializing classifier parameters: {e}")
        sys.exit(1)


def process_single_file(
    input_path: Union[str, Path], 
    output_path: Union[str, Path], 
    beta: np.ndarray, 
    beta_bias: np.ndarray,
    classes: list,  # New parameter for the list of class names
    threshold: Union[float, list] = 0.0,
    save_empty: bool = True
):
    """
    Process a single parquet file
    @param input_path: Path to the input parquet file
    @param output_path: Path to save the output csv file
    @param beta: Classifier weights as a numpy array
    @param beta_bias: Classifier bias as a numpy array
    @param classes: List of class names for classification
    @param threshold: Classification threshold (float for all classes, or list of floats for threshold per class) (default 0.0)
    @param save_empty: Whether to save empty results (default True)
    @return: True if processing was successful, False otherwise
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    metadata_columns = ['source', 'channel', 'offset']

    #num_feature_cols = 1280

    try:
        table = pq.read_table(input_path)
        
        print(f"Processing {input_path}: {table.num_rows} rows, {table.num_columns} columns")

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
        elif threshold is None:
            threshold_array = np.full(len(classes), np.finfo(np.float32).min, dtype=np.float32)
        else:
            raise TypeError("Threshold must be a float or a list of floats")
        
        above_threshold_mask = scores >= threshold_array
        # This gives us two 1D arrays: one for the row index, one for the class index
        passing_row_indices, passing_class_indices = np.where(above_threshold_mask)

        if len(passing_row_indices) == 0 and not save_empty:
            print("No scores passed the threshold. No output file will be created.")
            return True
        
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
        print(f"Saved results to {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def get_parquet_files(directory: Union[Path, str]) -> list:
    """Get all parquet files in directory recursively."""
    return list(Path(directory).rglob('*.parquet'))


def classify(input_path, output_path, config_path):
    """Main processing function for the 'classify' command."""

    config = load_config(config_path)
    classifier_config = config['classifier']
    beta, beta_bias = deserialize_classifier_params(classifier_config)
    
    print(f"Classes found in config: {classifier_config['classes']}")
    print(f"Beta shape: {beta.shape}, Beta bias shape: {beta_bias.shape}")

    threshold = config.get('threshold', 0.0)
    save_empty = config.get('save_empty', True)
    
    if input_path.is_file():
        if input_path.suffix != '.parquet':
            print("Error: Input file must be a parquet file", file=sys.stderr)
            sys.exit(1)
                
        success = process_single_file(
            input_path, output_path, beta, beta_bias, 
            classifier_config['classes'], threshold, save_empty
        )
        if success:
            print("Processing completed successfully")
        else:
            print("Processing failed")
            sys.exit(1)
    
    elif input_path.is_dir():

        parquet_files = get_parquet_files(input_path)
        if not parquet_files:
            print(f"No parquet files found in {input_path}")
            sys.exit(1)
        
        print(f"Found {len(parquet_files)} parquet files")
        
        success_count = 0
        for file_path in parquet_files:
            # Calculate relative path from input directory and construct output path
            rel_path = file_path.relative_to(input_path)
            output_file_path = output_path / rel_path.with_suffix('.csv') 

            success = process_single_file(
                file_path, output_file_path, beta, beta_bias, 
                classifier_config['classes'], threshold, save_empty
            )
                
            if success:
                success_count += 1
        
        print(f"Processing completed: {success_count}/{len(parquet_files)} files successful")
            
    else:
        print(f"Error: Input path '{input_path}' does not exist")
        sys.exit(1)


def show_version():
    """Reads and prints the content of the /VERSION file."""
    print("Running 'version' command...")
    version_file_container = Path('/VERSION')
    version_file_src = Path(__file__).parent / 'VERSION'

    if version_file_container.exists():
        version_file = version_file_container
    elif version_file_src.exists():
        version_file = version_file_src
    else:
        print("Error: /VERSION file not found in expected locations.", file=sys.stderr)
        sys.exit(1)
    version = version_file.read_text().strip()
    print(f"Version: {version}")


def using_container():
    """
    Check if the script is running inside a container by checking if the 
    /VERSION exists and matches the source VERSION file.
    """
    if not Path('/VERSION').exists():
        return False
    if Path('/VERSION').read_text().strip() != (Path(__file__).parent / 'VERSION').read_text().strip():
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
        classify(input_path, output_path, config_path)
    elif args.command == 'version':
        show_version()


if __name__ == "__main__":
    main()