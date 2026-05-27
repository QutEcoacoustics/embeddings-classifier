import argparse
import sys
import subprocess
from pathlib import Path
from typing import Union, Tuple
import shlex
import os
from dotenv import load_dotenv, find_dotenv

def sys_command(command: Union[str, list], cwd: Union[str, Path] = None) -> Tuple[str, str]:
    """
    Execute a system command and return the output (stdout, stderr).
    If command is a string, it will be split into a list (using shlex.split for safety).
    Generates a bash-friendly string for easy copy-pasting.
    """
    # Ensure command is a list of strings
    if isinstance(command, str):
        command_list = shlex.split(command)
    else:
        # Convert Path objects in the list to strings for subprocess
        command_list = [str(arg) for arg in command]
        
    command_for_bash = ' '.join([shlex.quote(arg) for arg in command_list])
    
    # Ensure cwd is a string if it's a Path object for subprocess.run
    cwd_str = str(cwd) if isinstance(cwd, Path) else cwd
    print(f"Executing command:\n {command_for_bash} \n in {cwd_str if cwd_str else 'current directory'}")

    process = None
    try:
        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd_str,
        )
        stdout, stderr = process.communicate()
    except KeyboardInterrupt as e:
        if process is not None:
            print("KeyboardInterrupt received. Terminating child process...")
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                print("Child process did not exit in time. Killing child process...")
                process.kill()
                process.communicate()
        raise RuntimeError("Execution interrupted by user (Ctrl+C).") from e
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command not found: '{command_list[0]}'. Please ensure it's in your PATH.") from exc
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while executing command: {e}") from e

    if process is not None and process.returncode != 0:
        full_error_output = (
            f"Command failed with return code {process.returncode}:\n"
            f"--- Captured STDOUT from Docker Container ---\n{stdout.strip()}\n"
            f"--- Captured STDERR from Docker Container ---\n{stderr.strip()}\n"
            f"--- End of Docker Container Output ---"
        )
        raise RuntimeError(full_error_output)

    return stdout.strip(), stderr.strip()


def run_docker_container(
    input_file_path: Path,
    output_folder_path: Path, # Changed name to better reflect its purpose as a folder
    config_file_path: Path,
    docker_image: str = "qutecoacoustics/crane-linear-model-runner:1.0.0",
    classify_args: Union[list, None] = None
) -> Tuple[str, str]:
    """
    Executes a Docker container with specified input, output, and config files.

    Args:
        input_file_path: Path to the input parquet file on the host.
        output_folder_path: Path to the output *folder* on the host where results will be saved.
        config_file_path: Path to the configuration JSON file on the host.
        docker_image: The Docker image to use for processing.

    Returns:
        A tuple containing (stdout, stderr) from the docker command.

    Raises:
        FileNotFoundError: If input or config files do not exist.
        RuntimeError: If the docker command fails.
    """
    # Resolve paths to ensure they are absolute and canonical for mounting
    input_file_path = input_file_path.resolve()
    output_folder_path = output_folder_path.resolve()
    config_file_path = config_file_path.resolve()

    print(f"--- Container run initiated ---")
    print(f"Input file: {input_file_path}")
    print(f"Output folder: {output_folder_path}")
    print(f"Config file: {config_file_path}")
    print(f"Docker image: {docker_image}")

    if not input_file_path.exists():
        raise FileNotFoundError(f"Error: Input file not found at '{input_file_path}'")
    if not config_file_path.exists():
        raise FileNotFoundError(f"Error: Config file not found at '{config_file_path}'")

    if not output_folder_path.exists():
        print(f"Creating output directory: {output_folder_path}")
        output_folder_path.mkdir(parents=True, exist_ok=True)
    elif not output_folder_path.is_dir():
        # Edge case: if a file already exists with the output folder name
        raise ValueError(f"Output path '{output_folder_path}' exists but is not a directory.")

    # The container path for the input file needs to include its name, not just the directory
    # The container path for the config file is fixed to 'config.json' within '/mnt/config'
    # The output path in the container is the directory /mnt/output
    input_container_path = f"/mnt/input/{input_file_path.name}"
    config_container_path = f"/mnt/config/config.json"
    output_container_path = f"/mnt/output" # This should be the directory where the container writes files



    # Construct the docker run command
    # Note: Using Path objects directly in f-strings converts them to strings.
    docker_command = [
        "docker", "run", "--rm", # --rm removes container after exit
        "-v", f"{input_file_path}:{input_container_path}",
        "-v", f"{config_file_path}:{config_container_path}",
        "-v", f"{output_folder_path}:{output_container_path}"
    ]

    baw_auth_token = get_auth_token()
    print(f"BAW_AUTH_TOKEN: {baw_auth_token}")
    if baw_auth_token:
        qsp = f"user_token={baw_auth_token}"
        docker_command.extend(["-e", f"QSP={qsp}"])
    else:
        exit("Error: BAW_AUTH_TOKEN environment variable is not set.")

    docker_command.append(docker_image)

    if classify_args:
        docker_command.extend(classify_args)

    stdout, stderr = sys_command(docker_command)

    print(f"Container stderr output:\n{stderr}")
    print(f"Container stdout output:\n{stdout}")
    print(f"--- Container run finished ---")

    return stdout, stderr

def get_auth_token() -> str:
    baw_auth_token = os.getenv('BAW_AUTH_TOKEN')
    if not baw_auth_token:
        load_dotenv(find_dotenv())
    baw_auth_token = os.getenv('BAW_AUTH_TOKEN')
    if not baw_auth_token:
        raise EnvironmentError("BAW_AUTH_TOKEN environment variable is not set.")
    return baw_auth_token


def main_cli():
    """
    Parses command-line arguments and calls the main run_docker_container function.
    Handles script exit based on errors.
    """
    parser = argparse.ArgumentParser(
        description="Process a parquet input file based on a configuration, and produce a CSV output file using a Docker container."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input parquet file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output *folder* where the output will be saved. The container will write files into this folder."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the configuration JSON file."
    )

    parser.add_argument(
        "--image",
        type=str,
        default="qutecoacoustics/crane-linear-model-runner:1.0.0",
        help="Docker image to use for processing."
    )

    args = parser.parse_args()

    try:
        # Call the reusable function with parsed arguments
        stdout, stderr = run_docker_container(
            input_file_path=args.input,
            output_folder_path=args.output, # Pass the path object directly
            config_file_path=args.config,
            docker_image=args.image
        )
        # You can add checks here for stdout if needed for success/failure
        print("Script execution successful.")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main_cli()