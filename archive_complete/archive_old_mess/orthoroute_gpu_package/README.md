# OrthoRoute GPU Autorouter

Professional GPU-accelerated PCB autorouting plugin for KiCad.

## Features

- 🚀 **GPU Acceleration**: Uses CuPy for high-performance routing on NVIDIA GPUs
- 🔄 **CPU Fallback**: Automatically falls back to NumPy when GPU is not available
- 📐 **Orthogonal Routing**: Specialized in clean orthogonal track layouts
- 🎯 **Smart Pathfinding**: Advanced A* algorithm with GPU-optimized wave propagation
- 🔌 **No SWIG**: Modern KiCad 9.0+ IPC API integration
- 📦 **Easy Install**: Single ZIP file installation via KiCad Plugin Manager

## Installation

1. Download the `orthoroute-gpu-package.zip` file
2. In KiCad, go to Tools → Plugin and Content Manager
3. Click "Install from File" and select the ZIP file
4. The plugin will automatically install dependencies

## Requirements

- KiCad 9.0 or later
- Python 3.8+
- NVIDIA GPU with CUDA support (optional, will use CPU fallback)

### Dependencies (auto-installed)
- numpy >= 1.20.0
- cupy-cuda12x >= 12.0.0 (for GPU acceleration)

## Usage

1. Open your PCB in KiCad PCB Editor
2. Click the **OrthoRoute GPU** button in the toolbar
3. The plugin will analyze your board and perform GPU-accelerated routing
4. Progress and results will be displayed in the console

## Performance

- **GPU Mode**: Utilizes CUDA cores for parallel pathfinding
- **CPU Mode**: Optimized NumPy fallback for systems without GPU
- **Smart Grid**: Adaptive resolution based on board complexity

## Development

This plugin uses the modern KiCad IPC API for communication, avoiding legacy SWIG dependencies. The routing engine is implemented in pure Python with optional CuPy GPU acceleration.

## License

MIT License - see LICENSE file for details.
