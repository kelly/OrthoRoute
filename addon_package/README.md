# OrthoRoute KiCad Addon Package

This directory contains the KiCad addon package structure for OrthoRoute GPU Autorouter.

## Structure

```
addon_package/
├── metadata.json          # Package metadata for KiCad PCM
├── plugins/              # Plugin files
│   ├── __init__.py       # Main plugin with UI and routing logic
│   ├── orthoroute_engine.py  # Standalone GPU routing engine
│   └── icon.png          # 24x24 toolbar icon
└── resources/            # Package resources
    └── icon.png          # 64x64 package icon
```

## Building the Package

To create the installable zip package:

```bash
python build_addon.py
```

This creates `orthoroute-kicad-addon.zip` which can be installed via KiCad's Plugin and Content Manager.

## Installation

1. Open KiCad
2. Go to **Tools → Plugin and Content Manager**
3. Click **Install from File**
4. Select the `orthoroute-kicad-addon.zip` file
5. Follow the installation prompts

## Requirements

- KiCad 8.0+
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8+ or 12.x
- CuPy package: `pip install cupy-cuda12x`

## Features

- **GPU-accelerated routing**: Uses NVIDIA CUDA through CuPy for parallel algorithms
- **Real-time visualization**: Optional progress display during routing
- **Multi-layer support**: Works with complex PCB designs
- **Configurable parameters**: Grid pitch, iteration limits, via costs
- **Fallback mode**: Works without GPU for testing (CPU-only routing)

## Plugin Structure

The addon is designed as a self-contained package that doesn't require external Python packages to be installed in KiCad's environment. The routing engine is included directly in the plugin.

### Key Components

- **Main Plugin** (`__init__.py`): KiCad ActionPlugin interface with configuration dialog
- **Routing Engine** (`orthoroute_engine.py`): Standalone GPU routing implementation
- **Metadata** (`metadata.json`): Package information for KiCad PCM

This structure follows KiCad's official addon packaging guidelines and ensures reliable installation and loading.
