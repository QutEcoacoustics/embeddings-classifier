#!/usr/bin/env python3
from fileinput import filename
from pathlib import Path
import csv
import json
from datetime import datetime, timedelta
import collections
import re
import argparse

def process_datetime(iso_string, offset_seconds):
    """
    Add seconds to an ISO datetime string and return full datetime and hour-rounded datetime.
    
    Args:
        iso_string (str): ISO format datetime string (e.g., '2024-04-23T07:27:02.000Z')
        offset_seconds (int): Number of seconds to add (can be negative)
    
    Returns:
        tuple: (full_datetime_string, hour_datetime_string)
               Both with same timezone offset format
    """

    file_dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
    segment_dt = file_dt + timedelta(seconds=offset_seconds)
    return segment_dt.isoformat()


baw_pattern = re.compile(r'\d{8}T\d{6}(?:Z|[+-]\d{4})_(.+?)_(\d+)\.\w+$')

def baw_source(source, baw_instance="https://api.ecosounds.org"):
    if not baw_instance:
        return (source, None)
    match = baw_pattern.match(source)
    if match:
        site_name = match.group(1)
        arid = int(match.group(2))
        source = f"{baw_instance}/audio_recordings/{arid}/original"
        return (source, site_name)
    else:
        return (source, None)
    
def baw_listen_link(baw_instance, arid, start_offset, end_offset):
    """
    Generate a listen link for a BAW audio recording.
    
    Args:
        baw_instance (str): Base URL of the BAW instance
        arid (int): Audio recording ID
        start_offset (int): Start offset in seconds
        end_offset (int): End offset in seconds
    
    Returns:
        str: Listen link URL
    """

    website_map = {
        "https://api.ecosounds.org": "https://www.ecosounds.org",
        "https://api.acousticobservatory.org": "https://data.ecosounds.org"
    }

    if baw_instance in website_map:
        baw_instance = website_map[baw_instance]

    return f"{baw_instance}/listen/{arid}/?start={start_offset}&end={end_offset}"


def process_csv(csv_file: Path, filelist_dic, output_file: Path, baw_instance="https://api.ecosounds.org"):
    """
    reads the csv, finds the corresponding file in the (filename stem is the id) filelist
    adds metadata columns to the data
    writes the rows to the output path. 
    """

    try:
        arid = int(csv_file.stem)
    except ValueError:
        print(f"Invalid file name {csv_file.stem}, expected an integer ID.")
        return 0

    if arid not in filelist_dic:
        print(f"ID {arid} not found in filelist")
        return 0

    file_metadata = filelist_dic[arid]
    

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:

            full_datetime = process_datetime(file_metadata['recorded_date'], int(row['offset']))
            # source, site_name = baw_source(file_metadata['source'], baw_instance=baw_instance)
            end = int(row['offset']) + 5
            #listen_link = f"{baw_instance}/listen/{arid}/?start={int(row['offset'])}&end={end}"

            # add metadata columns
            processed_row = {}
            processed_row['audio_recording_id'] = arid
            processed_row['site_name'] = file_metadata['sites.name']
            processed_row['site_id'] = int(file_metadata['site_id'])
            processed_row['region_name'] = file_metadata['regions.name']
            processed_row['region_id'] = int(file_metadata['regions.id'])
            processed_row['end_offset_seconds'] = end
            processed_row['start_offset_seconds'] = int(row['offset'])
            processed_row['label'] = row['label']
            processed_row['score'] = float(row['score'])
            processed_row['start_datetime'] = full_datetime
            processed_row['listen_link'] = baw_listen_link(baw_instance, arid, int(row['offset']), end)
            rows.append(processed_row)
    if not rows:
        return 0

    # write the rows to the output path

    if not output_file.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()

    # append the rows
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writerows(rows)

    return len(rows)


def aggregate_and_split_results(results_file: Path, aggregated_file: Path, non_max_file: Path):
    """
    Reads a CSV, groups rows by site/label/hour, and splits the output.

    1.  aggregated_file: Contains the single row with the max score for each 
        group, plus a 'count' column.
    2.  non_max_file: Contains all other rows that were not the maximum 
        for their respective group.
    """
    grouped_rows = collections.defaultdict(list)
    total_rows = 0
    with open(results_file, 'r', newline='') as f_in:
        reader = csv.DictReader(f_in)
        input_header = reader.fieldnames
        for row in reader:
            hour = row['start_datetime'][:13]
            key = (row['site_name'], row['label'], hour)
            float(row['score']) 
            grouped_rows[key].append(row)
            total_rows += 1

    output_header_aggregated = input_header + ['count']

    essential_columns = ['audio_recording_id', 'label', 'start_offset_seconds', 'end_offset_seconds', 'score']
    compact_aggregated_file = aggregated_file.with_name(aggregated_file.stem + '_compact.csv')
    compact_non_max_file = non_max_file.with_name(non_max_file.stem + '_compact.csv')
    
    with open(aggregated_file, 'w', newline='') as f_agg, \
         open(non_max_file, 'w', newline='') as f_non_max, \
         open(compact_aggregated_file, 'w', newline='') as f_agg_compact, \
         open(compact_non_max_file, 'w', newline='') as f_non_max_compact:
         

        agg_writer = csv.DictWriter(f_agg, fieldnames=output_header_aggregated)
        non_max_writer = csv.DictWriter(f_non_max, fieldnames=input_header)
        agg_compact_writer = csv.DictWriter(f_agg_compact, fieldnames=essential_columns)
        non_max_compact_writer = csv.DictWriter(f_non_max_compact, fieldnames=essential_columns)
        
        agg_writer.writeheader()
        non_max_writer.writeheader()
        agg_compact_writer.writeheader()
        non_max_compact_writer.writeheader()


        for rows_in_group in grouped_rows.values():
            max_score_row = max(rows_in_group, key=lambda r: float(r['score']))
            output_row_agg = max_score_row.copy()
            output_row_agg['count'] = len(rows_in_group)
            agg_writer.writerow(output_row_agg)
            agg_compact_writer.writerow({col: output_row_agg[col] for col in essential_columns})
            for row in rows_in_group:
                if row is not max_score_row:
                    non_max_writer.writerow(row)
                    non_max_compact_writer.writerow({col: row[col] for col in essential_columns})

    unique_labels = set(group[0]['label'] for group in grouped_rows.values())
    unique_sites = set(group[0]['site_name'] for group in grouped_rows.values())

    print(f"finished processing {len(grouped_rows)} aggregated rows and {total_rows - len(grouped_rows)} non-max rows.")
    print(f"Unique labels: {unique_labels}, Num sites: {len(unique_sites)}")



def generate_missing_files_summary(filelist, results_files, output_file):
    """
    Generate a summary of missing results files grouped by site.
    
    Args:
        filelist (list): List of metadata dictionaries with 'id' keys
        results_files (list): List of Path objects for existing CSV files
        output_file (Path): Base output file path for naming the summary file
    
    Returns:
        dict: Summary dictionary with missing files information
    """
    # Track which results files exist
    existing_arids = set()
    for csv_file in results_files:
        try:
            arid = int(csv_file.stem)
            existing_arids.add(arid)
        except ValueError:
            continue

    # Find missing results files and group by site
    missing_by_site = collections.defaultdict(list)
    for item in filelist:
        arid = item['id']
        if arid not in existing_arids:
            missing_by_site[item['sites.name']].append({
                'arid': arid,
                'site_id': item['site_id'],
                'date': item['recorded_date'][:10]  # Extract date part
            })

    # Create missing files summary
    missing_summary = {
        "missing_results_total": sum(len(missing) for missing in missing_by_site.values()),
        "missing_results_summary": []
    }

    for site_name, missing_items in missing_by_site.items():
        # Sort by date
        missing_items.sort(key=lambda x: x['date'])
        
        site_summary = {
            "site_name": site_name,
            "site_id": missing_items[0]['site_id'],  # All items have same site_id
            "missing_count": len(missing_items),
            "missing_arids": [item['arid'] for item in missing_items],
            "missing_dates": sorted(list(set(item['date'] for item in missing_items)))
        }
        missing_summary["missing_results_summary"].append(site_summary)

    # Sort sites by missing count (descending)
    missing_summary["missing_results_summary"].sort(key=lambda x: x['missing_count'], reverse=True)

    # Save missing files summary
    missing_summary_file = output_file.with_name(output_file.stem + '_missing_file_summary.json')
    with open(missing_summary_file, 'w') as f:
        # Convert to JSON string with normal formatting, then fix the lists
        json_str = json.dumps(missing_summary, indent=2)
        
        # Replace multi-line arrays with single-line arrays
        import re
        # Pattern to match arrays that span multiple lines
        array_pattern = r'\[\s*\n\s*((?:\d+,?\s*\n\s*)*\d+)\s*\n\s*\]'
        
        def compress_array(match):
            # Extract numbers and join them on one line
            numbers = re.findall(r'\d+', match.group(1))
            return '[' + ', '.join(numbers) + ']'
        
        # Also handle string arrays (dates)
        string_array_pattern = r'\[\s*\n\s*((?:"[^"]+",?\s*\n\s*)*"[^"]+")\s*\n\s*\]'
        
        def compress_string_array(match):
            # Extract quoted strings and join them on one line
            strings = re.findall(r'"[^"]+"', match.group(1))
            return '[' + ', '.join(strings) + ']'
        
        json_str = re.sub(array_pattern, compress_array, json_str)
        json_str = re.sub(string_array_pattern, compress_string_array, json_str)
        
        f.write(json_str)
    
    print(f"Missing files summary saved to: {missing_summary_file}")
    print(f"Total missing results files: {missing_summary['missing_results_total']}")
    
    return missing_summary

def main(filelist, results_dir, output_file, limit=None):

    # read the filelist json
    filelist_path = Path(filelist)
    if not filelist_path.exists():
        raise FileNotFoundError(f"Filelist does not exist: {filelist_path}")
    
    with open(filelist_path, 'r') as f:
        filelist = json.load(f)

    if limit and limit > 0:
        filelist = filelist[:limit]

    # filelist is a list of dicts with an 'id' key
    # restructure it to a dict where the id is the key for quick lookup
    filelist_dict = {item['id']: item for item in filelist}

    # for each csv in output dir
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {results_dir}")
    
    results_files = list(results_dir.rglob("*.csv"))
    output_file = Path(output_file)
    
    # Generate missing files summary
    generate_missing_files_summary(filelist, results_files, output_file)

    total_rows = 0

    # remove any existing output file
    if output_file.exists():
        output_file.unlink()

    for i, csv_file in enumerate(results_files):
        total_rows += process_csv(csv_file, filelist_dict, output_file)
        if i % 100 == 0:
            print(f"Processed {i} of {len(results_files)} files, total rows: {total_rows}")

    print(f"Processed {len(results_files)} files, total rows: {total_rows}")

    aggregate_and_split_results(output_file, 
                                 output_file.with_name(output_file.stem + '_aggregated.csv'),
                                 output_file.with_name(output_file.stem + '_non_max.csv'))

    return total_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Postprocess detection results from EcoSounds by adding metadata and aggregating results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --filelist filelist.json --results_dir results/ --output_file output.csv
  %(prog)s --filelist /path/to/filelist.json --results_dir /path/to/results/ --output_file /path/to/output.csv
        """
    )
    
    parser.add_argument(
        "--filelist",
        required=True,
        help="Path to JSON filelist containing audio recording metadata. "
             "Each item should have an 'id' key matching CSV filenames."
    )
    
    parser.add_argument(
        "--results_dir", 
        required=True,
        help="Directory containing CSV files with detection results. "
             "CSV filenames should match audio recording IDs."
    )
    
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path for the consolidated output CSV file. "
             "Will also generate '_aggregated.csv' and '_non_max.csv' variants."
    )
    
    args = parser.parse_args()
    
    try:
        total_rows = main(args.filelist, args.results_dir, args.output_file)
        print(f"Successfully processed {total_rows} total detection rows.")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

