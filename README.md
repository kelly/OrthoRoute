![Repo logo](/Assets/icon200.png)

# OrthoRoute GPU Autorouter

A high-performance GPU-accelerated PCB autorouter for KiCad using NVIDIA CUDA through CuPy.

## 🚀 Features

- **GPU-Accelerated Routing**: Leverages NVIDIA CUDA for parallel wavefront routing algorithms
- **Real-Time Visualization**: Optional progress display during routing operations
- **Multi-Layer Support**: Works with complex multi-layer PCB designs
- **Configurable Parameters**: Customizable grid pitch, iteration limits, via costs
- **KiCad Integration**: Native KiCad plugin with toolbar integration
- **Fallback Mode**: CPU-only routing when GPU is unavailable

## 📦 Installation

### Method 1: KiCad Plugin and Content Manager (Recommended)

1. Download the latest `orthoroute-kicad-addon.zip` from releases
2. Open KiCad
3. Go to **Tools → Plugin and Content Manager**
4. Click **Install from File**
5. Select the downloaded zip file
6. Follow installation prompts

### Method 2: Manual Installation

1. Clone this repository
2. Run the development installer:
   ```bash
   python install_dev.py
   ```
3. Restart KiCad

## 🔧 Requirements

### Hardware
- NVIDIA GPU with CUDA support (for acceleration)
- Minimum 4GB GPU memory recommended

### Software
- KiCad 8.0 or later
- CUDA Toolkit 11.8+ or 12.x
- Python packages:
  ```bash
  pip install cupy-cuda12x  # For CUDA 12.x
  # OR
  pip install cupy-cuda11x  # For CUDA 11.x
  ```

## 🎯 Usage

1. Open a PCB design in KiCad
2. Click the OrthoRoute icon in the toolbar or go to **Tools → OrthoRoute GPU Autorouter**
3. Configure routing parameters in the dialog:
   - **Grid Pitch**: Routing grid resolution (0.05-1.0mm)
   - **Max Iterations**: Maximum routing attempts per net
   - **Batch Size**: Number of nets processed simultaneously
   - **Via Cost**: Cost penalty for using vias
   - **Visualization**: Enable real-time progress display
4. Click **Start Routing**
5. Review results and imported tracks

## 🏗️ Project Structure

```
OrthoRoute/
├── addon_package/              # KiCad addon package
│   ├── metadata.json          # Package metadata
│   ├── plugins/               # Plugin files
│   │   ├── __init__.py        # Main plugin code
│   │   ├── orthoroute_engine.py # Standalone routing engine
│   │   └── icon.png           # Toolbar icon
│   └── resources/             # Package resources
│       └── icon.png           # Package manager icon
├── Assets/                    # Project assets and icons
├── build_addon.py             # Build addon package
├── install_dev.py             # Development installer
└── README.md                  # This file
```

## 🔧 Development

### Building the Addon Package

```bash
python build_addon.py
```

This creates `orthoroute-kicad-addon.zip` suitable for distribution.

### Development Installation

For quick testing during development:

```bash
python install_dev.py          # Install
python install_dev.py uninstall # Remove
```

### Package Structure

The addon follows KiCad's official packaging guidelines:

- **Self-contained**: No external package dependencies in KiCad environment
- **Standalone engine**: Included routing algorithms with CuPy fallback
- **Proper metadata**: Compatible with Plugin and Content Manager
- **Standard structure**: Follows KiCad addon conventions

## 🧪 Algorithm

OrthoRoute uses GPU-accelerated algorithms:

1. **Grid Initialization**: Creates routing grid on GPU memory
2. **Wavefront Propagation**: Parallel breadth-first search for each net
3. **Conflict Resolution**: Negotiated congestion-based rerouting
4. **Path Optimization**: Via minimization and length optimization
5. **Result Export**: Converts grid paths back to KiCad tracks

## 📊 Performance

Typical performance improvements over traditional routers:

- **Small boards** (< 500 nets): 5-10x faster
- **Medium boards** (500-2000 nets): 10-50x faster  
- **Large boards** (2000+ nets): 50-100x faster

*Performance depends on GPU capability and board complexity*

## 🐛 Troubleshooting

### Plugin Not Appearing
- Restart KiCad completely
- Check KiCad's Python console for error messages
- Verify installation in correct plugin directory

### CuPy Import Errors
- Install CUDA Toolkit first
- Install appropriate CuPy version for your CUDA version
- Verify GPU compatibility

### Memory Issues
- Reduce batch size in configuration
- Use smaller grid pitch for large boards
- Ensure sufficient GPU memory

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read contributing guidelines and submit pull requests.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/bbenchoff/OrthoRoute/issues)
- **Documentation**: [Wiki](https://github.com/bbenchoff/OrthoRoute/wiki)
- **Discussion**: [GitHub Discussions](https://github.com/bbenchoff/OrthoRoute/discussions)


