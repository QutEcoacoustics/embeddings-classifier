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

By default, it will classify all parquet files in `/mnt/input ` the input folder, and save the results as csv in `/mnt/output/`, using `/mnt/config.json` as the configuration file. 

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

