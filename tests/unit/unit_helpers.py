import base64
import json
from pathlib import Path
import pytest
from typing import Union
import shutil
from functools import reduce

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pv
import pyarrow.compute as pc

class UnitTestHelpers:
    """Helper functions for test setup and teardown."""
    
    @staticmethod
    def read_table(file_path: Union[str, Path]):
        """
        Read a parquet file and return a PyArrow table.
        If the file does not exist, raise an error.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        
        
        if file_path.suffix.lower() == '.parquet':
            return pq.read_table(file_path)
        elif file_path.suffix.lower() == '.csv':
            return pv.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: '{file_path.suffix}'")

    @staticmethod
    def compare_predictions(output_table_path, expected_results_path, tolerance=1e-3):
        """
        Compare the predictions in the output table with expected results.
        The expected results should be a list of tuples (id, label, score).
        """

        output_data = UnitTestHelpers.read_table(output_table_path).to_pylist()
        expected_data = UnitTestHelpers.read_table(expected_results_path)

        # We are checking that for each row in the output table
        # the score matches the 'logits' column in the expected results, for the same 
        # source, offset and label. 

        # we could sort both tables by these values, and then compare the whole score column 
        # but for now let's just loop through the output table and check each row


        def get_row(table, conditions: dict):
            masks = [pc.equal(table[column], value) for column, value in conditions.items()]
            combined_mask = reduce(pc.and_, masks)
            filtered_table = table.filter(combined_mask)
            return filtered_table.to_pylist()

        # for each row
        match_on = {
            'source': 'source_id',
            'offset': 'offset',
            'channel': 'channel',
            'label': 'label'
        }

        for i, output_row in enumerate(output_data):

            filter_contions = { b: output_row[a] for a, b in match_on.items() if a in output_row and b in expected_data.column_names }
            matching_expected_row = get_row(expected_data, filter_contions)

            if not matching_expected_row:
                raise ValueError(f"No matching row found in expected results for output row {i}: {output_row}")
            elif len(matching_expected_row) > 1:
                raise ValueError(f"Multiple matching rows found in expected results for output row {i}: {output_row}")
            else:
                # check if they are the same
                # assert matching_expected_row['logits'] == output_row['score'],\
                #     f"Score mismatch for output row {i}: expected {matching_expected_row['logits']}, got {output_row['score']}"
                # check if the score is within the tolerance
                matching_expected_row = matching_expected_row[0]
                assert np.isclose(matching_expected_row['logits'], output_row['score'], atol=tolerance),\
                    f"Score mismatch for output row {i}: expected {matching_expected_row['logits']}, got {output_row['score']}"
            
            print(output_row)
            

    @staticmethod
    def random_source():
        """Generate a random source string for testing."""
        
        def random_path():
            """Generate a random file path-like with random number of parts (subdirs)."""
            return '/'.join(['path'] + [str(np.random.randint(1000)) for _ in range(np.random.randint(1, 5))])
        
        def random_url():
            """Generate a random URL-like string."""
            return f'https://example.com{random_path()}'
        
        # return a random choice between URL and path
        return np.random.choice([random_url(), random_path()])


    @staticmethod
    def create_sample_parquet(file_path: Path, num_rows: int = 100, num_features: int = 1280, seed: int = 42, include_channel: bool = True):
        """Create a sample parquet file for testing using pure PyArrow."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(seed)
        source_array = [UnitTestHelpers.random_source() for _ in range(num_rows)]

        # random offsets: probably not realistic but we want to make it work for non-consecutive offsets
        offset_array = np.random.randint(0, 1000, size=num_rows).astype(np.int32)
        
        # Add feature columns (columns 2-1281, i.e., 1280 features)
        feature_arrays = []
        feature_names = []
        
        for i in range(num_features):
            feature_data = np.random.randn(num_rows).astype(np.float32)
            feature_arrays.append(pa.array(feature_data))
            feature_names.append(f'feature_{i}')

        if include_channel:
            channel_values = [[[1,2][i % 2] for i in range(num_rows)]]  # Alternating 1s and 2s
            channel_header = ['channel']
        else:
            channel_values = []
            channel_header = []
        
        # Combine all arrays and names
        all_arrays = [source_array, offset_array] + channel_values + feature_arrays
        all_names = ['source', 'offset'] + channel_header + feature_names
        
        # Create PyArrow table
        table = pa.table(all_arrays, names=all_names)
        
        # Write to parquet
        pq.write_table(table, file_path)
        
        return table
    

    @staticmethod
    def create_sample_classifier(num_features: int = 1280, num_classes: int = 1):
        """Create a sample classifier file for testing."""

        # Create random weights and bias
        np.random.seed(42)  # For reproducible tests
        weights = np.random.randn(num_features, num_classes).astype(np.float32)
        bias = np.random.randn(num_classes).astype(np.float32)

        # Encode to base64
        weights_b64 = base64.b64encode(weights.tobytes()).decode('utf-8')
        bias_b64 = base64.b64encode(bias.tobytes()).decode('utf-8')

        classifier = {
            "classes": [f"class_{i}" for i in range(num_classes)],
            "beta": weights_b64,
            "beta_bias": bias_b64
        }
        
        return classifier

    @staticmethod
    def create_sample_config(file_path: Path = None, 
                             num_features: int = 1280, 
                             num_classes: int = 1,
                             name: str = False,
                             threshold: Union[float, None] = 0.0):
        """Create single config file for a single classifier (dict format)."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        classifier = UnitTestHelpers.create_sample_classifier(num_features, num_classes)
        
        config = {
            "classifier": classifier,
        }

        if threshold is not None:
            config["threshold"] = threshold

        if name is not False:
            # We may use None for the name to indicate no classifier name subfolder
            # so use false here to indicate creating default name
            config["classifier_name"] = name

        if file_path is not None:
            # file_path is optional because we may be creating a list config for multiple classifiers
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        return config

    @staticmethod
    def create_sample_config_list(file_path: Path, 
                                  config_params_list: list):
        
        """
        Creates a single config file for multiple classifiers (list format)
        """
        
        file_path.parent.mkdir(parents=True, exist_ok=True)

        config = [UnitTestHelpers.create_sample_config(**config_params) for config_params in config_params_list]

        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
            
