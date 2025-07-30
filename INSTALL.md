# OrthoRoute Installation Guide

## Quick Installation (Recommended)

### Method 1: KiCad Plugin Manager
1. Download `orthoroute-kicad-addon.zip` (49.2KB optimized package)
2. In KiCad PCB Editor, go to **Tools → Plugin and Content Manager**
3. Click **Install from File** and select the downloaded zip
4. Restart KiCad
5. The plugin appears under **Tools → External Plugins → OrthoRoute GPU Autorouter**

### Method 2: Manual Installation
1. Extract `orthoroute-kicad-addon.zip` to your KiCad plugins directory:
   - **Windows**: `%APPDATA%\kicad\8.0\3rdparty\plugins\`
   - **Linux**: `~/.local/share/kicad/8.0\3rdparty\plugins\`
   - **macOS**: `~/Library/Application Support/kicad/8.0\3rdparty/plugins/`
2. Restart KiCad

## Development Installation

### Prerequisites
- KiCad 7.0+ or 8.0+ (with KiCad 9.0+ IPC API support)
- Python 3.8+
- Git (for cloning repository)
- Optional: NVIDIA GPU with CUDA support

### Clone and Install
```bash
# Clone the repository
git clone https://github.com/bbenchoff/OrthoRoute.git
cd OrthoRoute

# Install in development mode
python install_dev.py
```

The development installer:
- Copies plugin files to KiCad plugins directory
- Sets up symbolic links for live development
- Installs required Python dependencies
- Validates installation

### Build from Source
```bash
# Build optimized package
python build_addon.py
# Then install the generated zip via Plugin Manager
```

## API Compatibility

OrthoRoute supports both current and future KiCad APIs:

- **SWIG API (pcbnew)**: KiCad 7.0-8.0 compatibility
- **IPC API (kicad-python)**: KiCad 9.0+ future support
- **Automatic Detection**: Seamlessly switches between APIs

## Verification

### Test Plugin Installation
```bash
# Run verification script
python development/testing/verify_plugin.py

# Run comprehensive tests
python development/testing/run_all_tests.py
```

### Test in KiCad
1. Open KiCad PCB Editor
2. Load a PCB with unrouted nets
3. Go to **Tools → External Plugins → OrthoRoute GPU Autorouter**
4. Configure parameters and click **Route Board**

## Recent Improvements (2025)
- ✅ **64% Size Reduction**: Package optimized from 137.3KB to 49.2KB
- ✅ **Fixed Core Functionality**: Plugin crashes, missing track creation, net-pad matching
- ✅ **Future-Proof API Support**: Both SWIG and IPC API compatibility
- ✅ **Comprehensive Testing**: Headless testing with KiCad CLI integration
- ✅ **Professional Structure**: Clean organization with development files separated

## System Requirements
- **KiCad 7.0+ or 8.0+** (with KiCad 9.0+ IPC API support)
- **Python 3.8+** (bundled with KiCad)
- **Cross-Platform**: Windows, Linux, macOS support
- **Optional GPU**: NVIDIA GPU with CUDA support for acceleration

**Installation complete! OrthoRoute is production-ready and optimized for performance.** 🚀

## Verification
After installation, the OrthoRoute icon should appear in the KiCad PCB Editor toolbar. The plugin can be accessed via:
- Toolbar icon (🔀)
- Tools → External Plugins → OrthoRoute GPU Autorouter
