#!/usr/bin/env python3
"""
Parquet file processor that applies classification model to data.

Processes single files or directories of parquet files, applies a linear
classifier (beta weights + bias), and saves results maintaining folder structure.
"""

import argparse
import json
import base64
import os
import sys
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process parquet files with classification model"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input parquet file or directory containing parquet files"
    )
    parser.add_argument(
        "--output", "-o", 
        required=True,
        help="Output file or directory for results"
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="JSON configuration file with classifier parameters"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        if 'classifier' not in config:
            raise ValueError("Config must contain 'classifier' section")
        
        classifier = config['classifier']
        required_fields = ['classes', 'beta', 'beta_bias']
        for field in required_fields:
            if field not in classifier:
                raise ValueError(f"Classifier must contain '{field}' field")
        
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
    try:
        # Decode beta weights
        beta_bytes = base64.b64decode(classifier_config['beta'])
        beta = np.frombuffer(beta_bytes, dtype=np.float32)
        
        # Decode beta bias
        beta_bias_bytes = base64.b64decode(classifier_config['beta_bias'])
        beta_bias = np.frombuffer(beta_bias_bytes, dtype=np.float32)
        
        return beta, beta_bias
    
    except Exception as e:
        print(f"Error deserializing classifier parameters: {e}")
        sys.exit(1)


def process_single_file(input_path: str, output_path: str, beta: np.ndarray, beta_bias: np.ndarray):
    """Process a single parquet file."""
    try:
        # Read parquet file
        df = pd.read_parquet(input_path)
        print(f"Processing {input_path}: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Extract columns 3-1282 (0-indexed: columns 2-1281)
        if df.shape[1] < 1282:
            print(f"Warning: File has only {df.shape[1]} columns, expected at least 1282")
            feature_cols = min(df.shape[1] - 2, 1280)  # Adjust if fewer columns
        else:
            feature_cols = 1280
        
        # Get feature data (columns 2 to 2+feature_cols-1)
        features = df.iloc[:, 2:2+feature_cols].values.astype(np.float32)
        
        # Ensure beta matches feature dimensions
        if len(beta) != features.shape[1]:
            print(f"Error: Beta dimension ({len(beta)}) doesn't match features ({features.shape[1]})")
            return False
        
        # Apply classifier: features @ beta + beta_bias
        predictions = features @ beta + beta_bias
        
        # Create results dataframe
        results_df = df.iloc[:, :2].copy()  # Keep first 2 columns
        results_df['prediction'] = predictions
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_parquet(output_path, index=False)
        print(f"Saved results to {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def get_parquet_files(directory: str) -> list:
    """Get all parquet files in directory recursively."""
    parquet_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    return parquet_files


def main(input_path: str, output_path: str, config_path: str):
    """Main processing function."""
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Config: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    classifier_config = config['classifier']
    
    print(f"Classes: {classifier_config['classes']}")
    
    # Deserialize classifier parameters
    beta, beta_bias = deserialize_classifier_params(classifier_config)
    print(f"Beta shape: {beta.shape}, Beta bias shape: {beta_bias.shape}")
    
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    
    # Check if input is file or directory
    if os.path.isfile(input_path):
        # Single file processing
        if not input_path.endswith('.parquet'):
            print("Error: Input file must be a parquet file")
            sys.exit(1)
        
        # Ensure output has .parquet extension
        if not output_path.endswith('.parquet'):
            output_path += '.parquet'
        
        success = process_single_file(input_path, output_path, beta, beta_bias)
        if success:
            print("Processing completed successfully")
        else:
            print("Processing failed")
            sys.exit(1)
    
    elif os.path.isdir(input_path):
        # Directory processing
        parquet_files = get_parquet_files(input_path)
        
        if not parquet_files:
            print(f"No parquet files found in {input_path}")
            sys.exit(1)
        
        print(f"Found {len(parquet_files)} parquet files")
        
        # Process each file maintaining directory structure
        successful = 0
        for file_path in parquet_files:
            # Calculate relative path from input directory
            rel_path = os.path.relpath(file_path, input_path)
            output_file_path = os.path.join(output_path, rel_path)
            
            if process_single_file(file_path, output_file_path, beta, beta_bias):
                successful += 1
        
        print(f"Processing completed: {successful}/{len(parquet_files)} files successful")
        
        if successful == 0:
            sys.exit(1)
    
    else:
        print(f"Error: Input path '{input_path}' does not exist")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.input, args.output, args.config)