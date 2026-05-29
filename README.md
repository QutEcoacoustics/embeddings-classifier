# Embeddings Classifier

Container to run linear model with embeddings input and classification output using json config to load model weights

## Install from source

The application is installable as a Python package.

Install directly from a local checkout:

`pip install -e .`

Install from git source in another project:

`pip install "git+https://github.com/<org>/<repo>.git"`

Then import from Python:

`import embeddings_classifier.app`

## Python API (In-Memory)

The package now supports classifying in-memory data directly, with optional writing to disk.

### Classify an Arrow table (in-memory only)

```python
import pyarrow as pa
from embeddings_classifier import classify_table

# table must include metadata columns: source, channel, offset
table = pa.table({
  "source": ["file_a.wav"],
  "channel": [1],
  "offset": [0.0],
  "feature_0": [0.1],
  # ... remaining feature columns
})

results = classify_table(table, "./config.json")

for r in results:
  if r.success and r.result_table is not None:
    print(r.result_table.num_rows)
```

### Classify an Arrow table and also write output files

```python
from pathlib import Path
from embeddings_classifier import classify_table

results = classify_table(
  table,
  "./config.json",
  output_path=Path("./outputs/result.csv"),
)

# If output_path includes '<classifier_name>', it will be replaced per classifier.
# Otherwise, the same output path is used directly.
```

### Classify a pandas DataFrame

```python
import pandas as pd
from embeddings_classifier import classify_dataframe

df = pd.DataFrame({
  "source": ["file_a.wav"],
  "channel": [1],
  "offset": [0.0],
  "feature_0": [0.1],
  # ... remaining feature columns
})

results = classify_dataframe(df, "./config.json")
```

Notes:

- `classify_table` and `classify_dataframe` return a list of `ClassifierResult` (one per classifier).
- Each `ClassifierResult` may include `result_table` (in-memory output), `output_path`, `success`, and error/message info.
- If `output_path` is `None`, no files are written.


# Getting started

## Running tests

There are two types of tests: unit tests, which are best run from inside the container, and integration tests that are run from the host and test that the container does the correct thing when passed the parameters.

## Integration tests

Run `pytest -m docker tests/docker` to run tests from the host. This will launch the docker container. You will need the docker daemon running. 

**This requires that you have pytest installed in your python environment on the host.**


## Unit tests

Interactively: 

Run `docker run -ti --rm qutecoacoustics/crane-linear-model-runner sh` to launch the container interactively, then `python -m pytest -m "not docker" tests/unit ` to run the tests that are not the docker tests. 

From host directly: 

Run `docker run --rm --entrypoint python lmr:latest -m pytest -m "not docker" tests/unit`

## Usage

To run:

`docker run --rm -v input_folder:/mnt/input -v output_folder:/mnt/output -v config_file:/mnt/config/config.json <image_name>`

When running the provided Docker image, defaults are supplied through environment variables in the image (`EMBEDDINGS_CLASSIFIER_INPUT`, `EMBEDDINGS_CLASSIFIER_OUTPUT`, `EMBEDDINGS_CLASSIFIER_CONFIG`).

CLI path resolution for `classify` uses this precedence:

1. Explicit CLI args: `--input`, `--output`, `--config`
2. Environment variables: `EMBEDDINGS_CLASSIFIER_INPUT`, `EMBEDDINGS_CLASSIFIER_OUTPUT`, `EMBEDDINGS_CLASSIFIER_CONFIG`

If a path is missing from both args and environment, the command exits with an error.

# Configuration file

A configuration json file must be supplied. 
The configuration file contains:

- classifier: the model, including 
  - the list of classes, 
  - the beta (weights per class)
    - this is am ascii base64 encoded numpy array of shape (num_classes, embedding_size) 
  - beta_bias
    - this is am ascii base64 encoded numpy array of shape (num_classes, ) 
  - any model config saved with the model (not implemented yet but this will be included with the results)
- threshold: the cutoff for the output 

Examples can be found in `tests/test_data/config`

## Build package artifacts (local)

Wheel/release publishing is not configured yet, but local package builds can be produced with:

`python -m pip install --upgrade build`

`python -m build`

Artifacts will be created in `dist/`.

