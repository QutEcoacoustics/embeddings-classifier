import argparse
import json
from pathlib import Path
import time
from collections import deque
import logging
import logging.handlers

import pandas as pd

from recognizer_workshop.baw_api import baw_api
from recognizer_workshop import baw_helpers

from run_container import run_docker_container

api = None

def safe_name(name, id):
    """
    - convert to lowercase
    - replace spaces with underscores
    - trim
    - append id 
    """
    return f"{name.lower().replace(' ', '_').strip()}_{id}"




def get_filelist(baw_filter, filelist_path):

    logging.info(f"Fetching filelist with filter: {baw_filter}")
    
    if filelist_path.exists():
        with open(filelist_path, "r") as f:
            filelist = json.load(f)
        logging.info(f"Filelist read from cache: {filelist_path}")
        return filelist

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

    return filelist


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


def process_file(file, recognizers, output_path, timing_store):


    parquet_path = Path(output_path) / 'parquet_temp' / f"{file['id']}.parquet"
         

    for recognizer_name, config_path in recognizers.items():
        logging.info(f"Using recognizer: {recognizer_name}")

        recognizer_output_path = Path(output_path) / 'outputs' / recognizer_name
        site_output_path = recognizer_output_path / safe_name(file['sites.name'], file['id'])
        results_path = site_output_path / parquet_path.with_suffix('.csv').name
        site_output_path.mkdir(parents=True, exist_ok=True)

        if results_path.exists():
            logging.info(f"Results already exist for {file['id']} with recognizer {recognizer_name}, skipping...")
            continue

        download_successful = get_parquet(file['id'], parquet_path, timing_store)
        if not download_successful:
            logging.error(f"Download failed for {file['id']}. Skipping processing this file")
            return

        # --- Time Container Run ---
        start_time = time.perf_counter()
        run_docker_container(
            input_file_path=parquet_path,
            output_folder_path=site_output_path,
            config_file_path=Path(config_path)
        )
        container_duration = time.perf_counter() - start_time
        timing_store['container_run_times'].append(container_duration)
        avg_container_time = sum(timing_store['container_run_times']) / len(timing_store['container_run_times'])
        logging.info(
            f"Container for {file['id']} finished in {container_duration:.2f}s "
            f"(rolling 10-average: {avg_container_time:.2f}s)"
        )
        # read results csv and add metadata columns

        if results_path.exists():
            df = pd.read_csv(results_path)

            df['audio_recording_id'] = file['id']
            df['end_time_seconds'] = df['offset'] + 5.0
            df = df.rename(columns={"offset": "start_time_seconds"})

            # save
            df.to_csv(results_path, index=False)
            logging.info(f"Results saved to {results_path}")
        else:
            logging.error(f"Failed to produce results for {file['id']}, file not found at {results_path}")
    
    try:
        if parquet_path.exists():
            logging.info(f"Deleting temporary parquet file: {parquet_path}")
            parquet_path.unlink()
    except OSError as e:
        logging.error(f"Error deleting file {parquet_path}: {e}")

            
        


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


def main(params_path, limit=-1):

    global api
    api = baw_api()


    with open(params_path, "r") as f:
        params = json.load(f)

    if isinstance(params, dict):
        params = [params]

    for i, run in enumerate(params):
        output_dir = Path(run['output'])
        output_dir.mkdir(parents=True, exist_ok=True)

        timing_store = setup_logging(output_dir)

        filelist_path = Path(run['output']) / "filelist.json"

        filelist = get_filelist(run['filter'], filelist_path)

        logging.info(f"{len(filelist)} files found")

        if limit > 0:
            logging.info(f"Limiting to {limit} files out of {len(filelist)} total.")
            filelist = filelist[:limit]  # Limit the number of files for testing

        for i, file in enumerate(filelist):
            logging.info(f"--- Processing file {i+1}/{len(filelist)} (ID: {file['id']}) for run {i+1}/{len(params)} ---")
            process_file(file, run['recognizers'], run['output'], timing_store=timing_store)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run model on ecosounds data.")
    parser.add_argument("--params", type=Path, required=True, help="Path to JSON params for running on ecosounds data")
    parser.add_argument("--limit", type=int, default=-1, help="Limit the number of files to process.")
    args = parser.parse_args()

    main(args.params, args.limit)


