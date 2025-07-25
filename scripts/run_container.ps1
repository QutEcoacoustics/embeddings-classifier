#
# Executes a Docker container with a specified input file or folder, output folder, and config file.
#
param(
    [Parameter(Mandatory=$true)]
    [string]$InputPath, # Renamed from $input to avoid conflict with PowerShell's automatic variable

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
    # This robustly gets the absolute paths for the provided arguments.
    $InputAbs = Resolve-Path -Path $InputPath # Using the renamed variable
    $ConfigFileAbs = Resolve-Path -Path $ConfigFile
    $OutputFolderAbs = Convert-Path -Path $OutputFolder

    # --- Path Validation ---
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
        $InputFileName = [System.IO.Path]::GetFileName($InputAbs)
        $inputVolumeMount = "${InputAbs}:/mnt/input/${InputFileName}"
        Write-Host "Input is a file. Mounting as: $inputVolumeMount"
    }
    else {
        $inputVolumeMount = "${InputAbs}:/mnt/input"
        Write-Host "Input is a folder. Mounting as: $inputVolumeMount"
    }


    # --- Execute Docker Command ---
    Write-Host "Executing Docker command..."

    $dockerArgs = @(
        "run", "--rm",
        "-v", $inputVolumeMount,
        "-v", "${ConfigFileAbs}:/mnt/config/config.json",
        "-v", "${OutputFolderAbs}:/mnt/output",
        $Image
    )
    
    docker @dockerArgs

    Write-Host "--- Container run finished ---"
    Write-Host "Script execution successful."

} catch {
    Write-Error "Execution failed: $_"
    exit 1
}