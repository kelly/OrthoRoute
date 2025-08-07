# OrthoRoute GPU Autorouter

High-performance GPU-accelerated PCB autorouter for KiCad 9.0+

## Features

- 🚀 **GPU Acceleration**: Uses CuPy for NVIDIA CUDA parallel processing
- 🔄 **CPU Fallback**: Automatic fallback when GPU unavailable  
- 📐 **Smart Routing**: Advanced A* pathfinding with wave propagation
- 🎯 **Real-time Feedback**: Live visualization in KiCad GUI
- 🔌 **Modern API**: KiCad 9.0+ IPC integration (no SWIG)
- 📦 **Easy Install**: Plugin and Content Manager compatible

## Installation

### Via Plugin and Content Manager (Recommended)
1. Download `orthoroute-gpu-1.0.0.zip`
2. In KiCad, go to **Tools → Plugin and Content Manager**
3. Click **"Install from File..."**
4. Select the ZIP file and click **Install**
5. Restart KiCad

### Manual Installation
1. Extract ZIP contents to:
   - Windows: `%HOMEPATH%\Documents\KiCad\9.0\plugins\orthoroute-gpu\`
   - Linux: `~/.local/share/kicad/9.0/plugins/orthoroute-gpu\`
   - macOS: `~/Documents/KiCad/9.0/plugins/orthoroute-gpu\`
2. Restart KiCad

## Usage

1. **Enable API Server**: Go to **Preferences → Plugins → "Enable external plugin API server"**
2. **Open PCB**: Load your board in PCB Editor
3. **Click Button**: Look for **"Run GPU Autorouter"** button in toolbar
4. **Watch Magic**: See real-time routing with GPU acceleration!

## Requirements

- KiCad 9.0 or later
- Python 3.8+
- NVIDIA GPU with CUDA (optional - will use CPU fallback)

### Dependencies (auto-installed)
- `kicad-python` - KiCad IPC API bindings
- `cupy-cuda12x` - GPU acceleration (optional)
- `numpy` - Mathematical operations

## Performance

- **GPU Mode**: Massively parallel pathfinding on CUDA cores
- **CPU Mode**: Optimized A* algorithm for systems without GPU
- **Real-time**: Live visualization as routing progresses

## Architecture

This plugin uses KiCad 9.0's modern IPC (Inter-Process Communication) API:
- **Out-of-process**: Plugin runs in separate Python environment
- **Stable**: Plugin crashes won't affect KiCad
- **Modern**: No legacy SWIG dependencies
- **Extensible**: Easy to add new routing algorithms

## License

MIT License - see LICENSE file for details
