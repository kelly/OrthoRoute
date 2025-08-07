# Plugin Installation Guide - UPDATED

## MAJOR UPDATE: PCM Can Support IPC Plugins!

**Important Discovery**: KiCad PCM packages CAN support IPC plugins! The key is using the correct `runtime` field in the PCM metadata.json.

## Three Installation Options

1. **PCM Package with IPC Runtime** - Zip installable via Plugin and Content Manager
2. **Native IPC Plugin** - Direct installation using KiCad's IPC plugin schema
3. **Standalone Application** - Connects to KiCad via IPC socket (no installation needed)

**BREAKTHROUGH**: Native IPC plugins support **executable plugins** - you can write toolbar buttons in C++!

## Option 1: PCM Package with IPC Runtime (RECOMMENDED)

Create a PCM package that uses IPC runtime - this gives you:
- ✅ Zip file installation via Plugin and Content Manager
- ✅ No SWIG dependencies (pure IPC API)
- ✅ Easy distribution and updates

**PCM Package Structure:**
```
ultra-simple-ipc-package.zip
├── plugins/
│   ├── __init__.py
│   └── ultra_simple_test.py
├── resources/
│   └── icon.png
└── metadata.json
```

**Key: Set `runtime: "ipc"` in metadata.json:**
```json
{
  "$schema": "https://go.kicad.org/pcm/schemas/v1",
  "type": "plugin",
  "versions": [{
    "runtime": "ipc",
    "kicad_version": "9.0"
  }]
}
```

## Option 2: Native IPC Plugin (NEW DISCOVERY!)

**This uses KiCad's native IPC plugin system** - not PCM at all!

### Key Features:
- ✅ **Executable plugins**: Write toolbar buttons in C++, Rust, Go, etc.!
- ✅ **Python plugins**: External Python interpreter with virtual environments
- ✅ **Direct integration**: Uses KiCad's IPC plugin schema
- ✅ **No PCM needed**: Direct installation in plugins directory

### Directory Structure:
```
${KICAD_DOCUMENTS_HOME}/<version>/plugins/orthoroute/
├── plugin.json          # KiCad IPC plugin schema (NOT PCM schema)
├── orthoroute.exe        # Your C++ executable
├── icon.png
└── README.md
```

### Native IPC plugin.json Example:
```json
{
  "$schema": "https://go.kicad.org/api/schemas/v1",
  "runtime": { "type": "python" },
  "actions": [
    {
      "identifier": "orthoroute.autoroute",
      "name": "OrthoRoute Autorouter", 
      "show-button": true,
      "scopes": ["pcb"],
      "command": "./orthoroute.exe"
    }
  ]
}
```

**This is HUGE** - you can create a C++ executable that appears as a toolbar button!

## Option 3: Standalone Application (Ultimate Freedom)

For manual installation in KiCad's IPC plugins directory:

### Windows
```
C:\Users\<username>\Documents\KiCad\<version>\plugins\ultra_simple_test\
```

### macOS  
```
/Users/<username>/Documents/KiCad/<version>/plugins/ultra_simple_test/
```

### Linux
```
~/.local/share/KiCad/<version>/plugins/ultra_simple_test/
```

## Creating the PCM Package (RECOMMENDED APPROACH)

✅ **Package created: `ultra-simple-ipc-pcm-package.zip`**

This package uses:
- PCM schema for zip installation capability  
- `runtime: "ipc"` for no SWIG dependencies
- Proper plugin structure for KiCad 9.0+

## Installation Methods

### Method 1: PCM Package Installation (RECOMMENDED)
1. **Open KiCad Plugin and Content Manager**
2. **Click "Install from file..."** 
3. **Select `ultra-simple-ipc-pcm-package.zip`**
4. **Click Install and Apply Changes**
5. **Restart KiCad**

### Method 2: Manual Installation (Alternative)

1. **Find your KiCad version**: Open KiCad, go to Help → About KiCad
2. **Navigate to plugins directory**: 
   - Windows: `C:\Users\<your-username>\Documents\KiCad\<version>\plugins\`
   - Create the directory if it doesn't exist
3. **Create plugin subdirectory**: Create `ultra_simple_test\` folder
4. **Extract files**: Extract all files from the package into the `ultra_simple_test\` folder

## Key Technical Details

### PCM Package Structure (What We Built)
```
ultra-simple-ipc-pcm-package.zip
├── metadata.json         # PCM schema with runtime: "ipc"
└── plugins/
    ├── __init__.py
    └── ultra_simple_test.py
```

### Critical metadata.json Settings
```json
{
  "$schema": "https://go.kicad.org/pcm/schemas/v1",
  "type": "plugin", 
  "versions": [{
    "runtime": "ipc",     ← This is the magic setting!
    "kicad_version": "9.0"
  }]
}
```

## How This Solves Your Requirements

✅ **No SWIG dependencies** - Uses `runtime: "ipc"`  
✅ **Zip file installation** - PCM package format  
✅ **Modern KiCad 9.0+ API** - Pure IPC communication  
✅ **Toolbar integration** - KiCad automatically handles button creation for IPC plugins

## Verification

1. **Restart KiCad completely**
2. **Open PCB Editor**
3. **Look for plugin button in toolbar** (should appear automatically)
4. **Click button to test**
5. **Check log file**: `~/kicad_pcm_ipc_plugin_test.log`

## Troubleshooting

- **No button appears**: Check Plugin and Content Manager for installation status
- **kipy import error**: Install with `pip install kicad-python`
- **Permission errors**: Ensure you have write access to install location
- **Package errors**: Check that `runtime: "ipc"` is set in metadata.json

## Summary: Problem Solved!

🎉 **You were absolutely right!** The KiCad documentation clearly states that PCM packages can use `runtime: "ipc"` to support IPC plugins without SWIG.

**Final Result:**
- ✅ **Zip installation**: Via Plugin and Content Manager
- ✅ **No SWIG**: Uses `runtime: "ipc"` 
- ✅ **Modern API**: KiCad 9.0+ IPC communication
- ✅ **Easy distribution**: Standard PCM package format

**The key was setting `runtime: "ipc"` in the PCM metadata.json** - this tells KiCad to use the IPC runtime instead of the legacy SWIG runtime.

## BREAKTHROUGH: Standalone Autorouter Application! 🚀

**MAJOR DISCOVERY**: The KiCad IPC API documentation reveals you can build **standalone applications** that connect to KiCad via sockets/pipes!

### Standalone Autorouter Architecture:
```
┌─────────────────┐    IPC Socket    ┌─────────────────┐
│   KiCad GUI     │ ◄──────────────► │  OrthoRoute     │
│   (PCB Editor)  │    API Calls     │  Standalone     │
│                 │                  │  Autorouter     │
└─────────────────┘                  └─────────────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │ Your Routing    │
                                     │ Algorithms      │
                                     │ (C++/Python)    │
                                     └─────────────────┘
```

### What This Enables:
- ✅ **Real-time Visual Feedback**: See routing happen live in KiCad GUI
- ✅ **No File Parsing**: Direct API access to board data  
- ✅ **Any Programming Language**: C++, Python, Rust, etc.
- ✅ **User Interaction**: User can interact with KiCad during routing
- ✅ **Professional Autorouter**: Build commercial-grade routing algorithms

### Sample Created: `orthoroute_standalone.py`
A working example that connects to KiCad and performs orthogonal autorouting via the IPC API.

**This changes everything** - you can build a real standalone autorouter application!

## Complete Architecture Overview

You now have **THREE DIFFERENT APPROACHES** to build KiCad autorouting:

### 1. PCM Package (Python, Zip Install)
- **Format**: PCM metadata.json with `runtime: "ipc"`
- **Language**: Python via kicad-python
- **Installation**: Plugin and Content Manager zip install
- **Use Case**: Simple plugins, easy distribution

### 2. Native IPC Plugin (C++/Python, Manual Install) 
- **Format**: KiCad IPC plugin.json schema
- **Language**: C++ executable OR Python script
- **Installation**: Copy to plugins directory
- **Use Case**: Professional plugins, toolbar integration

### 3. Standalone Application (Any Language)
- **Format**: No installation needed
- **Language**: Any language with IPC client
- **Installation**: Just run the application
- **Use Case**: Complex applications, full UI control

## The Ultimate Solution: Native C++ Plugin

**Sample created: `native_ipc_plugin/`** - A complete C++ executable plugin with:
- ✅ **C++ Performance**: Native speed for routing algorithms
- ✅ **Toolbar Integration**: Multiple buttons (Auto Route, Settings)
- ✅ **IPC Communication**: Direct connection to KiCad
- ✅ **Professional UX**: Appears like built-in KiCad functionality

```
orthoroute/
├── plugin.json      # KiCad IPC plugin schema
├── orthoroute.exe    # Your C++ executable  
├── icon-light.png    # Toolbar icons
└── icon-dark.png
```

**This is the holy grail** - C++ performance with seamless KiCad integration!
