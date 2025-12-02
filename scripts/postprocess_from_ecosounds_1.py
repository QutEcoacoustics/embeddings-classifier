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


def process_csv(csv_file: Path, filelist_dic, output_file: Path, baw_instance="https://api.ecosounds.org", exclude_classes=None):
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
            # Skip excluded classes
            if exclude_classes and row['label'] in exclude_classes:
                continue

            full_datetime = process_datetime(file_metadata['recorded_date'], int(row['offset']))
            # source, site_name = baw_source(file_metadata['source'], baw_instance=baw_instance)
            end = int(row['offset']) + 5
            #listen_link = f"{baw_instance}/listen/{arid}/?start={int(row['offset'])}&end={end}"

            # add metadata columns
            processed_row = {}
            processed_row['audio_recording_id'] = arid
            processed_row['site_name'] = file_metadata['sites.name']
            processed_row['site_id'] = int(file_metadata['site_id'])
            if file_metadata.get('regions.name'):
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


def aggregate_and_split_results(results_file: Path, max_per_hour_file: Path, non_max_file: Path, exclude_classes=None):
    """
    Reads a CSV, groups rows by site/label/hour, and splits the output.

    1.  max_per_hour_file: Contains the single row with the max score for each 
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
            # Skip excluded classes
            if exclude_classes and row['label'] in exclude_classes:
                continue
            
            hour = row['start_datetime'][:13]
            key = (row['site_name'], row['label'], hour)
            float(row['score']) 
            grouped_rows[key].append(row)
            total_rows += 1

    output_header_max_per_hour = input_header + ['count']

    # Add site_id to essential_columns for compact outputs
    essential_columns = ['audio_recording_id', 'site_id', 'label', 'start_offset_seconds', 'end_offset_seconds', 'score']
    compact_max_per_hour_file = max_per_hour_file.with_name(max_per_hour_file.stem + '_compact.csv')
    compact_non_max_file = non_max_file.with_name(non_max_file.stem + '_compact.csv')

    # Prepare data for splitting if needed
    max_per_hour_rows = []
    non_max_rows = []
    max_per_hour_compact_rows = []
    non_max_compact_rows = []

    for rows_in_group in grouped_rows.values():
        max_score_row = max(rows_in_group, key=lambda r: float(r['score']))
        output_row_agg = max_score_row.copy()
        output_row_agg['count'] = len(rows_in_group)
        max_per_hour_rows.append(output_row_agg)
        # Add site_id to compact row
        max_per_hour_compact_rows.append({
            'audio_recording_id': output_row_agg['audio_recording_id'],
            'site_id': output_row_agg['site_id'],
            'label': output_row_agg['label'],
            'start_offset_seconds': output_row_agg['start_offset_seconds'],
            'end_offset_seconds': output_row_agg['end_offset_seconds'],
            'score': output_row_agg['score'],
        })
        for row in rows_in_group:
            if row is not max_score_row:
                non_max_rows.append(row)
                non_max_compact_rows.append({
                    'audio_recording_id': row['audio_recording_id'],
                    'site_id': row['site_id'],
                    'label': row['label'],
                    'start_offset_seconds': row['start_offset_seconds'],
                    'end_offset_seconds': row['end_offset_seconds'],
                    'score': row['score'],
                })

    # save a version of the max_per_hour with only the top 1000 scores
    max_per_hour_top_1000_rows = sorted(max_per_hour_rows, key=lambda r: float(r['score']), reverse=True)[:1000]
    max_per_hour_top_1000_file = max_per_hour_file.with_name(max_per_hour_file.stem + '_top_1000.csv')


    # Helper to split and save by site if needed, saving in subfolders
    def save_csv_split_if_needed(rows, header, base_file: Path, site_key='site_id', subfolder=None):
        if len(rows) > 100000:
            # Split by site, save in subfolder
            site_rows = collections.defaultdict(list)
            for row in rows:
                site = row[site_key]
                site_rows[site].append(row)
            folder = base_file.parent / (subfolder if subfolder else base_file.stem)
            folder.mkdir(parents=True, exist_ok=True)
            for site, site_rows_list in site_rows.items():
                site_file = folder / f"{base_file.stem}_site_{site}{base_file.suffix}"
                with open(site_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writeheader()
                    writer.writerows(site_rows_list)
        else:
            if subfolder:
                folder = base_file.parent / subfolder
                folder.mkdir(parents=True, exist_ok=True)
                file_path = folder / base_file.name
            else:
                file_path = base_file
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)

    # Save max_per_hour (was aggregated)
    save_csv_split_if_needed(max_per_hour_rows, output_header_max_per_hour, max_per_hour_file, site_key='site_id', subfolder='max_per_hour')
    save_csv_split_if_needed(non_max_rows, input_header, non_max_file, site_key='site_id', subfolder='non_max')
    save_csv_split_if_needed(max_per_hour_compact_rows, essential_columns, compact_max_per_hour_file, site_key='site_id', subfolder='max_per_hour_compact')
    save_csv_split_if_needed(non_max_compact_rows, essential_columns, compact_non_max_file, site_key='site_id', subfolder='non_max_compact')

    save_csv_split_if_needed(max_per_hour_top_1000_rows, output_header_max_per_hour, max_per_hour_top_1000_file, site_key='site_id', subfolder='max_per_hour')

    unique_labels = set(group[0]['label'] for group in grouped_rows.values())
    unique_sites = set(group[0]['site_name'] for group in grouped_rows.values())

    print(f"finished processing {len(grouped_rows)} max_per_hour rows and {len(non_max_rows)} non-max rows.")
    print(f"Unique labels: {unique_labels}, Num sites: {len(unique_sites)}")



def generate_missing_files_summary(filelist, results_files, output_file, detection_counts=None):
    """
    Generate a summary of missing results files grouped by site.
    
    Args:
        filelist (list): List of metadata dictionaries with 'id' keys
        results_files (list): List of Path objects for existing CSV files
        output_file (Path): Base output file path for naming the summary file
        detection_counts (dict): Detection counts by site and class
    
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

    # Add detection counts to the summary if provided
    if detection_counts:
        missing_summary["detection_counts"] = detection_counts

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

def count_detections_from_csv_files(aggregated_compact_file, non_max_compact_file):
    """
    Count detections by class and site from the compact CSV files.
    
    Args:
        aggregated_compact_file (Path): Path to aggregated compact CSV
        non_max_compact_file (Path): Path to non-max compact CSV
    
    Returns:
        dict: Detection counts by total and by site
    """
    detection_counts = {
        "total_by_class": collections.defaultdict(int),
        "by_site": collections.defaultdict(lambda: collections.defaultdict(int))
    }
    
    # Read both compact files
    for csv_file in [aggregated_compact_file, non_max_compact_file]:
        if csv_file.exists():
            with open(csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # We need to get site info from the full filelist using audio_recording_id
                    # This will be handled in the main function where we have access to filelist_dict
                    pass
    
    return detection_counts

def split_and_save_full_results(results_file: Path, compact_file: Path, site_key='site_id', subfolder='results_by_site', compact_subfolder='results_compact_by_site'):
    """
    Split the full results CSV and its compact version by site and save each to a subfolder, but also keep the full CSVs.
    """
    # Full results
    with open(results_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames

    if not rows:
        return

    # Save compact version of full results
    essential_columns = ['audio_recording_id', 'site_id', 'label', 'start_offset_seconds', 'end_offset_seconds', 'score']
    compact_rows = [
        {col: row[col] for col in essential_columns}
        for row in rows
    ]
    with open(compact_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=essential_columns)
        writer.writeheader()
        writer.writerows(compact_rows)

    # Split full results by site if needed
    if len(rows) > 100000:
        site_rows = collections.defaultdict(list)
        for row in rows:
            site = row[site_key]
            site_rows[site].append(row)
        folder = results_file.parent / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        for site, site_rows_list in site_rows.items():
            site_file = folder / f"{results_file.stem}_site_{site}{results_file.suffix}"
            with open(site_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(site_rows_list)

    # Split compact results by site if needed
    if len(compact_rows) > 100000:
        site_rows = collections.defaultdict(list)
        for row in compact_rows:
            site = row[site_key]
            site_rows[site].append(row)
        folder = compact_file.parent / compact_subfolder
        folder.mkdir(parents=True, exist_ok=True)
        for site, site_rows_list in site_rows.items():
            site_file = folder / f"{compact_file.stem}_site_{site}{compact_file.suffix}"
            with open(site_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=essential_columns)
                writer.writeheader()
                writer.writerows(site_rows_list)

def split_csv_by_site_if_needed(csv_file: Path, site_key='site_id', subfolder=None, threshold=100000):
    """
    If the CSV has more than `threshold` rows, split it by site into a subfolder.
    """
    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames

    if not rows or len(rows) <= threshold:
        return

    site_rows = collections.defaultdict(list)
    for row in rows:
        site = row[site_key]
        site_rows[site].append(row)

    folder = csv_file.parent / (subfolder if subfolder else csv_file.stem)
    folder.mkdir(parents=True, exist_ok=True)
    for site, site_rows_list in site_rows.items():
        site_file = folder / f"{csv_file.stem}_site_{site}{csv_file.suffix}"
        with open(site_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(site_rows_list)

def main(filelist, results_dir, output_file, limit=None, exclude_classes=None):
    filelist_path = Path(filelist)
    if not filelist_path.exists():
        raise FileNotFoundError(f"Filelist does not exist: {filelist_path}")
    
    with open(filelist_path, 'r') as f:
        filelist = json.load(f)



    # filelist is a list of dicts with an 'id' key
    # restructure it to a dict where the id is the key for quick lookup
    filelist_dict = {item['id']: item for item in filelist}

    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {results_dir}")
    
    results_files = list(results_dir.rglob("*.csv"))
    output_file = Path(output_file)

    # subset for testing
    if limit and limit > 0:
        results_files = results_files[:limit]
    
    generate_missing_files_summary(filelist, results_files, output_file)

    total_rows = 0

    # remove any existing output file
    if output_file.exists():
        output_file.unlink()

    for i, csv_file in enumerate(results_files):
        total_rows += process_csv(csv_file, filelist_dict, output_file, exclude_classes=exclude_classes)
        if i % 100 == 0:
            print(f"Processed {i} of {len(results_files)} files, total rows: {total_rows}")

    print(f"Processed {len(results_files)} files, total rows: {total_rows}")

    # Save and split full results and its compact version
    results_compact_file = output_file.with_name(output_file.stem + '_compact.csv')
    # Save compact version of full results
    essential_columns = ['audio_recording_id', 'site_id', 'label', 'start_offset_seconds', 'end_offset_seconds', 'score']
    with open(output_file, 'r', newline='') as f_in, open(results_compact_file, 'w', newline='') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=essential_columns)
        writer.writeheader()
        for row in reader:
            writer.writerow({col: row[col] for col in essential_columns})

    max_per_hour_file = output_file.with_name(output_file.stem + '_max_per_hour.csv')
    non_max_file = output_file.with_name(output_file.stem + '_non_max.csv')
    aggregate_and_split_results(output_file, max_per_hour_file, non_max_file, exclude_classes=exclude_classes)

    compact_max_per_hour_file = output_file.with_name(output_file.stem + '_max_per_hour_compact.csv')
    compact_non_max_file = output_file.with_name(output_file.stem + '_non_max_compact.csv')

    # Now split all 6 files if needed
    split_csv_by_site_if_needed(output_file, site_key='site_id', subfolder='results_by_site')
    split_csv_by_site_if_needed(results_compact_file, site_key='site_id', subfolder='results_compact_by_site')
    split_csv_by_site_if_needed(max_per_hour_file, site_key='site_id', subfolder='max_per_hour')
    split_csv_by_site_if_needed(non_max_file, site_key='site_id', subfolder='non_max')
    split_csv_by_site_if_needed(compact_max_per_hour_file, site_key='site_id', subfolder='max_per_hour_compact')
    split_csv_by_site_if_needed(compact_non_max_file, site_key='site_id', subfolder='non_max_compact')

    # Count detections from the compact CSV files
    detection_counts = {
        "total_by_class": collections.defaultdict(int),
        "by_site": collections.defaultdict(lambda: collections.defaultdict(int))
    }
    
    # Read both compact files and count detections
    for csv_file in [compact_max_per_hour_file, compact_non_max_file]:
        if csv_file.exists():
            with open(csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    arid = int(row['audio_recording_id'])
                    label = row['label']
                    
                    # Skip excluded classes
                    if exclude_classes and label in exclude_classes:
                        continue
                    
                    # Get site name from filelist
                    if arid in filelist_dict:
                        site_name = filelist_dict[arid]['sites.name']
                        detection_counts["total_by_class"][label] += 1
                        detection_counts["by_site"][site_name][label] += 1
    
    # Convert defaultdicts to regular dicts for JSON serialization
    detection_counts["total_by_class"] = dict(detection_counts["total_by_class"])
    detection_counts["by_site"] = {site: dict(counts) for site, counts in detection_counts["by_site"].items()}
    
    # Regenerate summary with detection counts
    generate_missing_files_summary(filelist, results_files, output_file, detection_counts)

    return total_rows

main(
    filelist="./local_files/inference_results/fsp/results/filelist.json",
    results_dir="./local_files/inference_results/fsp/results/classifier_0",
    output_file="./local_files/inference_results/fsp/results/classifier_0_aggregated/fsp_dec25_results_01.csv"
)


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
    
    parser.add_argument(
        "--exclude_class",
        required=False,
        default=None,
        help="Comma-separated list of classes to exclude from results and counts."
    )
    
    args = parser.parse_args()
    
    # Parse exclude classes
    exclude_classes = None
    if args.exclude_class:
        exclude_classes = set(cls.strip() for cls in args.exclude_class.split(','))
        print(f"Excluding classes: {exclude_classes}")
    
    try:
        total_rows = main(args.filelist, args.results_dir, args.output_file, exclude_classes=exclude_classes)
        print(f"Successfully processed {total_rows} total detection rows.")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

