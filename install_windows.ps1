# Windows Installation Script for OrthoRoute
Write-Host "Installing OrthoRoute for Windows..." -ForegroundColor Green

# Check if running with admin privileges
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Please run this script as Administrator" -ForegroundColor Red
    exit
}

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "Python not found. Please install Python 3.8 or higher first." -ForegroundColor Red
    exit
}

# Check CUDA installation
Write-Host "Checking CUDA installation..." -ForegroundColor Yellow
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    Write-Host "NVIDIA drivers/CUDA not found. Please install CUDA Toolkit 11.8+ first." -ForegroundColor Red
    exit
}

# Create and activate virtual environment
Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install CuPy
Write-Host "Installing CuPy (this may take a while)..." -ForegroundColor Yellow
$cudaVersion = nvidia-smi --query-gpu=driver_version --format=csv,noheader
if ([version]$cudaVersion -ge [version]"12.0") {
    pip install cupy-cuda12x
} else {
    pip install cupy-cuda11x
}

# Install OrthoRoute
Write-Host "Installing OrthoRoute..." -ForegroundColor Yellow
pip install -e .

# Install KiCad Plugin
Write-Host "Installing KiCad plugin..." -ForegroundColor Yellow

# Try both possible plugin locations for KiCad 9.0
$PLUGIN_DIRS = @(
    "$env:USERPROFILE\Documents\KiCad\9.0\scripting\plugins\OrthoRoute",
    "$env:APPDATA\kicad\9.0\scripting\plugins\OrthoRoute"
)

$installed = $false
foreach ($PLUGIN_DIR in $PLUGIN_DIRS) {
    Write-Host "Attempting to install to: $PLUGIN_DIR"
    try {
        if (-not (Test-Path $PLUGIN_DIR)) {
            New-Item -ItemType Directory -Path $PLUGIN_DIR -Force
        }
        Copy-Item -Path "kicad_plugin\*" -Destination $PLUGIN_DIR -Recurse -Force
        $installed = $true
        Write-Host "Successfully installed plugin to: $PLUGIN_DIR" -ForegroundColor Green
        break
    }
    catch {
        Write-Host "Could not install to $PLUGIN_DIR" -ForegroundColor Yellow
        continue
    }
}

if (-not $installed) {
    Write-Host "Failed to install plugin to any location. Please check permissions." -ForegroundColor Red
    Write-Host "You can manually copy the kicad_plugin folder to one of these locations:" -ForegroundColor Yellow
    foreach ($dir in $PLUGIN_DIRS) {
        Write-Host $dir
    }
}

Write-Host "`nInstallation Complete!" -ForegroundColor Green
Write-Host "`nNext steps:"
Write-Host "1. Restart KiCad if it's running"
Write-Host "2. In KiCad PCB Editor, go to Tools > External Plugins > OrthoRoute GPU Autorouter"
Write-Host "3. If plugin doesn't appear, check Plugin and Content Manager"
Write-Host "`nTo test the installation, run:"
Write-Host "python -c 'import cupy; print(cupy.cuda.runtime.getDeviceCount())'"
Write-Host "This should show the number of available GPUs"
