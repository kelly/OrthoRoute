# OrthoRoute GPU Autorouter

High-performance GPU-accelerated PCB autorouter for KiCad 9.0+ using the modern IPC API.

## Features

- **GPU Acceleration**: Leverages NVIDIA CUDA through CuPy for parallel routing algorithms
- **Process Isolation**: Runs as separate process for maximum KiCad stability  
- **IPC API Integration**: Uses KiCad's modern IPC API (no SWIG dependencies)
- **Real-time Progress**: Visual feedback during routing operations
- **Multi-layer Support**: Handles complex multi-layer PCB designs

## Requirements

- **KiCad 9.0+** with IPC API support
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** 11.8+ or 12.x
- **Python 3.8+**

## Installation

### Option 1: Plugin and Content Manager (Recommended)

1. Download the latest `orthoroute-gpu-x.x.x.zip` from releases
2. Open KiCad PCB Editor
3. Go to **Tools → Plugin and Content Manager**
4. Click **Install from File**
5. Select the downloaded ZIP file
6. **Enable API server**: Go to **Preferences → Plugins** and check **"Enable external plugin API server"**
7. **Restart KiCad**
8. Look for the **"Run GPU Autorouter"** button in the PCB Editor toolbar

### Option 2: Manual Build

```bash
git clone https://github.com/bbenchoff/OrthoRoute.git
cd OrthoRoute
python build_plugin.py
```

## Usage

1. Open your PCB design in KiCad PCB Editor
2. Click the **"Run GPU Autorouter"** button in the toolbar
3. The plugin will automatically:
   - Connect to KiCad via IPC API
   - Analyze unrouted nets
   - Route using GPU-accelerated algorithms
   - Display progress and results

## Project Structure

```
OrthoRoute/
├── src/                    # Plugin source code
│   ├── gpu_autorouter.py   # Main entry point
│   ├── gpu_router.py       # GPU routing engine
│   ├── plugin.json         # KiCad plugin configuration
│   ├── metadata.json       # PCM package metadata
│   └── requirements.txt    # Python dependencies
├── build/                  # Build output
├── assets/                 # Icons and images
├── docs/                   # Documentation
├── tests/                  # Test files
└── build_plugin.py         # Build script
```

## Development

### Building from Source

```bash
python build_plugin.py
```

This creates `build/orthoroute-gpu-x.x.x.zip` ready for installation.

### Dependencies

The plugin automatically installs these dependencies when installed via PCM:

- `kicad-python>=0.4.0` - KiCad IPC API bindings
- `cupy-cuda12x>=12.0.0` - GPU acceleration
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.5.0` - Visualization

## Troubleshooting

### Plugin Not Visible

1. Ensure **API server is enabled**: Preferences → Plugins → "Enable external plugin API server"
2. **Restart KiCad** completely after installation
3. Check KiCad version is **9.0+**

### CUDA Errors

1. Verify NVIDIA GPU with CUDA support
2. Install **CUDA Toolkit 11.8+** or **12.x**
3. Test with: `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"`

### Import Errors

1. Check that dependencies are installed in KiCad's Python environment
2. The plugin creates its own virtual environment automatically via PCM

## License

WTFPL - Do What The F*ck You Want To Public License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with your KiCad setup
5. Submit a pull request

## Support

- **Issues**: [GitHub Issues](https://github.com/bbenchoff/OrthoRoute/issues)
- **Documentation**: [Wiki](https://github.com/bbenchoff/OrthoRoute/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/bbenchoff/OrthoRoute/discussions)
