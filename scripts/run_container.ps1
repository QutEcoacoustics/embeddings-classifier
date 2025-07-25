#
# Executes a Docker container with a specified input file or folder, output folder, and config file.
#
param(
    [Parameter(Mandatory=$true)]
    [string]$input, # Can be a single file or a directory

    [Parameter(Mandatory=$true)]
    [string]$OutputFolder,

    [Parameter(Mandatory=$true)]
    [string]$ConfigFile,

    [string]$Image = "qutecoacoustics/crane-linear-model-runner:1.0.0"
)

# Stop script on any error
$ErrorActionPreference = "Stop"

try {
    # --- Prepare and Validate Paths ---
    # Resolve paths to their absolute form
    $InputAbs = (Resolve-Path -Path $input).Path
    $ConfigFileAbs = (Resolve-Path -Path $ConfigFile).Path
    $OutputFolderAbs = (Resolve-Path -Path (Join-Path -Path (Get-Location) -ChildPath $OutputFolder) -ErrorAction SilentlyContinue).Path

    if (-not $OutputFolderAbs) {
        $OutputFolderAbs = (Join-Path -Path (Get-Location).Path -ChildPath $OutputFolder)
    }

    # Validate that the input path and config file exist
    if (-not (Test-Path -Path $InputAbs)) {
        throw "Error: Input path not found at '$InputAbs'"
    }
    if (-not (Test-Path -Path $ConfigFileAbs -PathType Leaf)) {
        throw "Error: Config file not found at '$ConfigFileAbs'"
    }

    # Create output directory if it doesn't exist
    if (-not (Test-Path -Path $OutputFolderAbs -PathType Container)) {
        Write-Host "Creating output directory: $OutputFolderAbs"
        New-Item -Path $OutputFolderAbs -ItemType Directory -Force | Out-Null
    }

    Write-Host "--- Container run initiated ---"
    Write-Host "Input path: $InputAbs"
    Write-Host "Output folder: $OutputFolderAbs"
    Write-Host "Config file: $ConfigFileAbs"
    Write-Host "Docker image: $Image"

    # --- Determine Input Volume Mount ---
    $inputVolumeMount = ""
    if (Test-Path -Path $InputAbs -PathType Leaf) {
        # Input is a FILE
        $InputFileName = [System.IO.Path]::GetFileName($InputAbs)
        $inputVolumeMount = "${InputAbs}:/mnt/input/${InputFileName}"
        Write-Host "Input is a file. Mounting as: $inputVolumeMount"
    }
    else {
        # Input is a FOLDER
        $inputVolumeMount = "${InputAbs}:/mnt/input"
        Write-Host "Input is a folder. Mounting as: $inputVolumeMount"
    }


    # --- Execute Docker Command ---
    Write-Host "Executing Docker command..."

    # Use splatting for cleaner command arguments
    $dockerArgs = @(
        "run", "--rm",
        "-v", $inputVolumeMount,
        "-v", "${ConfigFileAbs}:/mnt/config/config.json",
        "-v", "${OutputFolderAbs}:/mnt/output",
        $Image
    )
    
    # Execute docker with the arguments
    docker @dockerArgs

    Write-Host "--- Container run finished ---"
    Write-Host "Script execution successful."

} catch {
    Write-Error "Execution failed: $_"
    exit 1
}