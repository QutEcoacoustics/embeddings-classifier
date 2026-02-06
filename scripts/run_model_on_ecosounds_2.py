"""
This script allows do one ore more "runs" of classification of ecosounds data.
A "run" is set of recordings, defined by a baw filter, and a set of recognizers to run on those recordings. 
"""



import concurrent.futures
import os
import argparse
import json
from pathlib import Path
import time
from collections import deque
import logging
import logging.handlers
import random

import pandas as pd

from baw_helpers.baw_api import baw_api
from baw_helpers import baw_helpers

from run_container import run_docker_container

DEFAULT_DOCKER_IMAGE = "qutecoacoustics/crane-linear-model-runner:1.0.1"

api = None

def safe_name(name, id):
    """
    - convert to lowercase
    - replace spaces with underscores
    - trim
    - append id 
    """
    return f"{name.lower().replace(' ', '_').strip()}_{id}"




def get_filelist(baw_filter, filelist_path, limit=-1):

    logging.info(f"Fetching filelist with filter: {baw_filter}")

    if limit > 0:
        filelist_path = filelist_path.with_name(filelist_path.stem + f"_limit_{limit}.json")
    
    if filelist_path.exists():
        with open(filelist_path, "r") as f:
            filelist = json.load(f)
        logging.info(f"Filelist read from cache: {filelist_path}")
        return filelist, filelist_path

    payload = {
        'filter': baw_filter,
        'projection': {
            'include': ['id', 'site_id', 'sites.name', 'regions.id', 'regions.name', 'recorded_date']
        }
    }

    url = "audio_recordings/filter?disable_paging=true"

    response = api.post(url, payload)

    if 'data' not in response:
        raise Exception(f"Failed to fetch filelist: {response.status_code} {response.text}")
    
    filelist = response['data']

    filelist_path.parent.mkdir(parents=True, exist_ok=True)

    # write the filelist to a local file
    with open(filelist_path, "w") as f:
        json.dump(filelist, f, indent=4)
        f.write("\n")
        logging.info(f"Filelist fetched from API and written to {filelist_path}")
        logging.info(f"Number of files: {len(filelist)}")

    return filelist, filelist_path


def get_input_links(fileslist_path, ajid=3, analysis_name='perch-embeddings_4', throttle=-1, limit=-1, check_exists=False):
    """
    Reads a JSON file containing a list of audio recordings and creates a list of links to their parquet files,
    and the appropriate output paths (site folder structure).
    saves to a _sources.json file in the same directory as the input filelist and returns the path to that file.
    """

    sources_path = fileslist_path.with_name(fileslist_path.stem + "_sources.json")
    if sources_path.exists():
        return sources_path

    if not fileslist_path.exists():
        raise FileNotFoundError(f"File list not found: {fileslist_path}")

    with open(fileslist_path, "r") as f:
        data = json.load(f)

    input_links = {
        'source': [],
        'output': []
    }

    missing_files = []

    # for each audio recording, we need to make an api call to get the parquet file link
    logging.info(f"Fetching parquet links for {len(data)} audio recordings from API...")
    for i, item in enumerate(data):

        # We can make an api request to get the analysis job metadata for this file to get the parquet link
        # or we can just construct the link directly, but it may return 404 if the file doesn't exist
        #'https://api.ecosounds.org/analysis_jobs/3/results/4717608/perch-embeddings_4/embeddings.parquet'
        if not check_exists:
            source = f"/analysis_jobs/{ajid}/results/{item['id']}/{analysis_name}/embeddings.parquet"
        else:
            aj_meta = api.get(f"/analysis_jobs/{ajid}/results/{item['id']}/{analysis_name}")
            try:
                source = aj_meta.get("data").get("children", [])[0]['path']
            except (IndexError, KeyError) as e:
                missing_files.append(item['id'])
                logging.error(f"Failed to get source path for {item['id']}: {e}")
                continue
        source = api.base_url + source
        if item.get("regions.name"):
            output = f"{safe_name(item['regions.name'], item['regions.id'])}/{safe_name(item['sites.name'], item['site_id'])}/{item['id']}.csv"
        else:
            output = f"{safe_name(item['sites.name'], item['site_id'])}/{item['id']}.csv"
        input_links['source'].append(source)
        input_links['output'].append(output)

        if throttle > 0:
            time.sleep(throttle)

        # put a dot every 10 items to show progress
        if i % 10 == 0:
            print('.', end='', flush=True)
        if i % 100 == 0:
            logging.info(f"Processed {i} of {len(data)} files...")
        if len(input_links['source']) >= limit > 0:
            logging.info(f"Reached limit of {limit} files, stopping.")
            break


    with open(sources_path, "w") as f:
        json.dump(input_links, f, indent=4)
        f.write("\n")
        logging.info(f"Parquet links written to {sources_path}")

    return sources_path


def get_parquet(arid, destination_filename, timing_store):

    if destination_filename.exists():
        logging.info(f"Parquet file already exists for {arid}, skipping download.")
        return True

    start_time = time.perf_counter()

    result = baw_helpers.get_embeddings(
        baw_api=api,
        analysis_job_id=3,
        audio_recording_id=arid,
        destination_file=destination_filename)
    
    download_duration = time.perf_counter() - start_time
    timing_store['download_times'].append(download_duration)
    avg_download_time = sum(timing_store['download_times']) / len(timing_store['download_times'])

    if result:
        logging.info(
            f"Downloaded {arid} to {destination_filename} in {download_duration:.2f}s "
            f"(rolling 10-average: {avg_download_time:.2f}s)"
        )
    else:
        logging.error(f"Failed to download parquet for {arid} to {destination_filename}")
    
    return result


def process_from_links(inputs_json, config_path, output_path, timing_store, docker_image):

    Path(output_path).mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    run_docker_container(
        input_file_path=inputs_json,
        output_folder_path=output_path,
        config_file_path=Path(config_path),
        docker_image=docker_image,
        classify_args=['classify', '--input', str(Path('/mnt/input/') / inputs_json.name)],
    )
    container_duration = time.perf_counter() - start_time
    timing_store['container_run_times'].append(container_duration)
    avg_container_time = sum(timing_store['container_run_times']) / len(timing_store['container_run_times'])
    logging.info(
        f"Container run finished in {container_duration:.2f}s "
    )
            
     
def setup_logging(output_dir):
    # Ensure the output directory exists for the log file
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / "processing_log.log"

    # Remove any existing handlers to prevent duplicate logs if called multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the minimum level for all handlers

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # 1. RotatingFileHandler for size-based rotation
    # maxBytes: The maximum size of the log file in bytes before it rotates.
    #           10 * 1024 * 1024 = 10 MB
    # backupCount: The number of backup log files to keep.
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file_path,
        maxBytes=20 * 1024 * 1024, # 10 MB per file
        backupCount=1000,             # Keep the 100 most recent log files
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. StreamHandler for console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Storing timing information for rolling averages
    timing_store = {
        'download_times': deque(maxlen=10),  # Store last 10 download times
        'container_run_times': deque(maxlen=10)  # Store last 10 container run times
    }

    return timing_store


def main(params_path, limit=-1, docker_image=DEFAULT_DOCKER_IMAGE):
    """
    Main function to run the model on ecosounds data.
    Reads parameters from a JSON file, fetches filelist, processes files in parallel,
    and runs the model using Docker.

    params json should contain information requires to 

    """

    global api
    api = baw_api()

    with open(params_path, "r") as f:
        params = json.load(f)

    if isinstance(params, dict):
        params = [params]

    for run_index, run in enumerate(params):
        output_dir = Path(run['output'])
        timing_store = setup_logging(output_dir)

        logging.info(f"--- Starting run {run_index + 1}/{len(params)} with output to {output_dir} ---")

        filelist_path = output_dir / "filelist.json"
        _, filelist_path = get_filelist(run['filter'], filelist_path)
        inputs_json = get_input_links(filelist_path, run['analysis_job_id'], run['analysis_name'], limit=limit)

        try:
            process_from_links(            
                inputs_json,
                Path(run['config']),
                Path(run['output']),
                timing_store,
                docker_image)


            logging.info(f"({run_index + 1}/{len(params)}) runs completed successfully.")
        except Exception as exc:
            logging.error(f"({run_index + 1}/{len(params)}) run failed with exception: {exc}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model on ecosounds data in parallel.")
    parser.add_argument("--params", type=Path, required=True, help="Path to JSON params for running on ecosounds data")
    parser.add_argument("--limit", type=int, default=-1, help="Limit the number of files to process.")
    parser.add_argument("--docker_image", type=str, default=DEFAULT_DOCKER_IMAGE, help="Docker image to use for processing.")
    args = parser.parse_args()

    main(args.params, args.limit, args.docker_image)
