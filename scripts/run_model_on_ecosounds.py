import json
from pathlib import Path

from recognizer_workshop.baw_api import baw_api
from recognizer_workshop import baw_helpers

from run_container import run_docker_container

import pandas as pd

import argparse




api = baw_api()


def safe_name(name, id):
    """
    - convert to lowercase
    - replace spaces with underscores
    - trim
    - append id 
    """
    return f"{name.lower().replace(' ', '_').strip()}_{id}"




def get_filelist(baw_filter, filelist_path):
    
    if filelist_path.exists():
        with open(filelist_path, "r") as f:
            filelist = json.load(f)
        print(f"Filelist read from {filelist_path}")
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
        print(f"Filelist written to {filelist_path}")
        print(f"Number of files: {len(filelist)}")

    return filelist


def get_parquet(arid, destination_filename):

    result = baw_helpers.get_embeddings(
        baw_api=api,
        analysis_job_id=3,
        audio_recording_id=arid,
        destination_file=destination_filename)
    
    return result


def process_file(file, recognizer_name, config_path, output_path):

    parquet_path = Path(output_path) / 'parquet_temp' / f"{file['id']}.parquet"
    site_output_path = Path(output_path) / 'outputs' / recognizer_name / safe_name(file['sites.name'], file['id'])
    results_path = site_output_path / parquet_path.with_suffix('.csv').name

    if results_path.exists():
        print(f"Results already exist for {file['id']}, skipping...")
        return
   
    result = get_parquet(file['id'], parquet_path)

    if result:
        print(f"Successfully downloaded {file['id']} to {parquet_path}")

        run_docker_container(
            input_file_path=parquet_path,
            output_folder_path=site_output_path,
            config_file_path=Path(config_path)
        )

        # read results csv and add metadata columns

        if results_path.exists():
            df = pd.read_csv(results_path)

            df['audio_recording_id'] = file['id']
            df['end_time_seconds'] = df['offset'] + 5.0
            df = df.rename(columns={"offset": "start_time_seconds"})

            # save
            df.to_csv(results_path, index=False)
            print(f"Results saved to {results_path}")
        else:
            print(f"Failed to produce results for {file['id']}, file not found at {results_path}")
            
            








def main(params_path, limit=-1):

    with open(params_path, "r") as f:
        params = json.load(f)

    if isinstance(params, dict):
        params = [params]

    for run in params:
    
        filelist_path = Path(run['output']) / "filelist.json"

        filelist = get_filelist(run['filter'], filelist_path)

        print(len(filelist), "files found")

        if limit > 0:
            print(f"Limiting to {limit} files")
            filelist = filelist[:limit]  # Limit the number of files for testing

        for i, file in enumerate(filelist):

            for recognizer in run['recognizers']:

                process_file(file, recognizer['name'], recognizer['config_path'], run['output'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run model on ecosounds data.")
    parser.add_argument("--params", type=Path, required=True, help="Path to JSON params for running on ecosounds data")
    parser.add_argument("--limit", type=int, default=-1, help="Limit the number of files to process.")
    args = parser.parse_args()

    main(args.params, args.limit)


