# KiCad IPC API Fixes Applied - August 4, 2025

## Summary of Changes

Based on the official KiCad IPC API documentation, the following critical fixes have been applied to align OrthoRoute with the proper IPC plugin architecture:

## ✅ Fixed Issues

### 1. **Plugin Architecture Restructure** 
- **Before**: Mixed SWIG ActionPlugin + IPC API (incorrect hybrid approach)
- **After**: Pure IPC plugin architecture using official KiCad IPC API
- **Impact**: Proper separation of concerns, follows KiCad's official plugin specification

### 2. **Entry Point Structure** 
- **Before**: ActionPlugin class-based entry points
- **After**: Standalone Python scripts with `main()` functions as entry points
- **Files Updated**: 
  - `orthoroute_ipc_plugin.py` - Main IPC plugin (proper structure)
  - `simple_ipc_test.py` - Simple IPC API test
  - `plugin.json` - Updated entry points to match IPC specification

### 3. **Communication Protocol**
- **Before**: File-based JSON communication only  
- **After**: **Primary**: Native IPC API via Protocol Buffers over Unix sockets, **Secondary**: JSON files for GPU server
- **Impact**: Uses official KiCad communication protocol for board operations

### 4. **Process Model**
- **Before**: Plugin directly accessing SWIG APIs
- **After**: Plugin connects to KiCad via IPC API, launches separate GPU server process
- **Impact**: True process isolation with official KiCad support

### 5. **API Integration**
- **Before**: Mixed API usage causing conflicts
- **After**: Clean separation:
  - **Plugin**: Uses `kipy` (kicad-python) for all KiCad operations
  - **GPU Server**: Standalone process with no KiCad dependencies
  - **Legacy**: SWIG wrapper for backward compatibility only

### 6. **Directory Structure**
- **Before**: Unclear plugin structure  
- **After**: Proper IPC plugin package structure:
  ```
  addon_package/
  ├── orthoroute_ipc_plugin.py    # Main IPC plugin entry point
  ├── simple_ipc_test.py          # IPC API validation test
  ├── plugin.json                 # IPC plugin definitions (fixed)
  ├── metadata.json               # Package metadata
  ├── plugins/                    # Legacy SWIG + utility plugins
  │   ├── __init__.py             # Legacy SWIG wrapper only
  │   └── orthoroute_standalone_server.py  # GPU server
  └── resources/                  # Package resources
  ```

### 7. **Documentation Updates**
- **README.md**: Updated architecture diagrams to show IPC API integration
- **Requirements**: Clarified `kicad-python` package requirement
- **Installation**: Added IPC API verification steps

## 🔧 Technical Details

### IPC Plugin Implementation
The main plugin (`orthoroute_ipc_plugin.py`) now properly:
- Connects to KiCad using `kipy.KiCad()` class
- Reads environment variables set by KiCad (`KICAD_API_SOCKET`, `KICAD_API_TOKEN`) 
- Uses Protocol Buffers for all KiCad communication
- Extracts board data via IPC API calls
- Applies routing results via IPC API track creation
- Follows official KiCad plugin lifecycle

### GPU Server Integration  
The GPU server remains isolated but now integrates properly:
- Plugin extracts board data via IPC API
- Server processes routing using CUDA/CuPy in isolation
- Plugin applies results back to KiCad via IPC API
- No direct KiCad dependencies in server process

### Environment Variables
Plugin now properly uses KiCad-provided environment variables:
- `KICAD_API_SOCKET` - Path to IPC socket
- `KICAD_API_TOKEN` - Authentication token
- Automatically set by KiCad when launching IPC plugins

## 🧪 Testing & Validation

### New Test Plugin
- `simple_ipc_test.py` - Validates basic IPC API functionality
- Creates one test track to verify API connection
- Available in Tools → External Plugins → "Simple IPC Test"

### Verification Steps
1. Install `kicad-python`: `pip install kicad-python`
2. Install updated package via KiCad Plugin Manager
3. Test IPC connection with "Simple IPC Test"
4. Run full "OrthoRoute GPU Autorouter" if test passes

## 📦 Package Status

**New Package**: `orthoroute-kicad-addon.zip` (150.0 KB)
- ✅ Proper IPC plugin structure
- ✅ Plugin.json with correct entry points  
- ✅ kicad-python dependency specified
- ✅ Backward compatibility maintained
- ✅ Ready for KiCad 9.0+ installation

## 🚀 Next Steps

1. **Install the updated package** in KiCad 9.0+
2. **Verify IPC API** works with "Simple IPC Test"
3. **Test full routing** with GPU acceleration
4. **Report any issues** with the new IPC integration

---

**Key Benefit**: OrthoRoute now uses KiCad's **official, supported plugin architecture** instead of reverse-engineered SWIG bindings, ensuring long-term compatibility and stability.
