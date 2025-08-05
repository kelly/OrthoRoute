# IPC Plugin Structure Fix - Complete Implementation

## Critical Issue Identified and Fixed

**Problem**: Plugin was using incorrect structure for KiCad 9.0+ IPC plugins
**Root Cause**: IPC plugins don't use `.register()` - they use direct class instantiation

## Key Structural Changes Made

### 1. **Removed All SWIG Registration**
```python
# OLD (SWIG style - WRONG for IPC):
class MyPlugin(pcbnew.ActionPlugin):
    def Run(self):
        # code here
MyPlugin().register()  # ❌ This doesn't work for IPC

# NEW (IPC style - CORRECT):
class MyIPCPlugin:
    def run(self):
        # code here

if __name__ == "__main__":
    plugin = MyIPCPlugin()
    plugin.run()  # ✅ This is the IPC pattern
```

### 2. **Restructured All Plugin Files**

#### **Main Plugin**: `orthoroute_ipc_plugin.py`
- ✅ **Class-based structure**: `OrthoRouteIPCPlugin` class
- ✅ **IPC connection handling**: `connect_to_kicad()` method
- ✅ **Main execution method**: `run()` method
- ✅ **Direct instantiation**: `plugin = OrthoRouteIPCPlugin(); plugin.run()`

#### **Test Plugins**: 
- ✅ **`simple_ipc_test.py`**: `SimpleIPCTestPlugin` class
- ✅ **`minimal_track_plugin.py`**: `MinimalTrackPlugin` class

### 3. **IPC Plugin Pattern Implementation**

**Correct Structure**:
```python
#!/usr/bin/env python3
import sys
from kipy import KiCad

class MyIPCPlugin:
    def __init__(self):
        self.kicad = None
        
    def run(self):
        # Connect to KiCad
        self.kicad = KiCad()
        
        # Do plugin work
        board = self.kicad.get_board()
        # ... plugin logic ...
        
        return 0  # Success

# IPC Pattern - Direct execution
if __name__ == "__main__":
    plugin = MyIPCPlugin()
    exit_code = plugin.run()
    # No sys.exit() - just return
```

### 4. **Updated plugin.json Configuration**

✅ **Proper PCM Schema**: Uses `https://go.kicad.org/pcm/schemas/v1`
✅ **Action Definitions**: Each plugin action properly defined
✅ **Entry Points**: Correct file references for each plugin
✅ **Menu/Toolbar Settings**: Proper visibility configuration

### 5. **Removed All Dangerous sys.exit() Calls**

**Why this matters**: `sys.exit()` kills the entire Python interpreter, which kills KiCad since it embeds Python.

✅ **Import Error Handling**: Uses flags instead of `sys.exit()`
✅ **Main Block Handling**: No `sys.exit()` in `__main__` blocks
✅ **Error Handling**: Returns error codes instead of terminating

## Files Updated

### **Primary Plugin Files** (Critical for visibility):
1. **`addon_package/orthoroute_ipc_plugin.py`** - Main plugin entry point
2. **`addon_package/simple_ipc_test.py`** - Simple test plugin
3. **`addon_package/minimal_track_plugin.py`** - Minimal test plugin
4. **`addon_package/plugin.json`** - Plugin configuration
5. **`addon_package/plugins/__init__.py`** - Removed SWIG registration

### **Package Status**:
- ✅ **Size**: 96.4 KB (increased due to proper structure)
- ✅ **Schema**: Valid PCM schema
- ✅ **Entry Points**: All using IPC pattern
- ✅ **No sys.exit()**: Safe completion handling

## How IPC Plugins Work in KiCad 9.0+

### **Discovery Process**:
1. KiCad scans plugin directories for `plugin.json` files
2. Reads action definitions from `plugin.json`
3. Creates menu entries based on action configuration
4. When user clicks menu item, KiCad launches Python script as subprocess
5. Script runs with `__name__ == "__main__"` and executes directly

### **Execution Flow**:
```
KiCad Menu Click
    ↓
KiCad launches: python plugin_script.py
    ↓
Script executes: if __name__ == "__main__":
    ↓
Plugin class instantiated and run() method called
    ↓
Plugin connects via IPC API and performs work
    ↓
Plugin completes and returns exit code
    ↓
KiCad receives completion notification
```

### **Key Differences from SWIG**:
- **SWIG**: Plugins run in KiCad's process space (dangerous)
- **IPC**: Plugins run as separate processes (safe isolation)
- **SWIG**: Direct memory access to KiCad internals
- **IPC**: Protocol-based communication over sockets
- **SWIG**: `.register()` method for discovery
- **IPC**: `plugin.json` file for discovery

## Installation and Testing

### **Installation**:
1. Use the updated `orthoroute-kicad-addon.zip` (96.4 KB)
2. Install via KiCad Plugin and Content Manager
3. Restart KiCad completely

### **Expected Behavior**:
- **Menu Items**: Should appear in Tools → External Plugins
- **Execution**: Plugins run as separate processes
- **Stability**: No KiCad crashes on completion
- **Logging**: Console output visible in KiCad

### **Troubleshooting**:
- **Not Visible**: Check KiCad version (needs 9.0+)
- **Import Errors**: Install `kicad-python` package
- **Connection Errors**: Enable IPC API in KiCad preferences

## Status Summary

✅ **Structure Fixed**: All plugins use proper IPC class-based pattern
✅ **Registration Fixed**: Removed SWIG `.register()`, using `plugin.json`
✅ **Execution Fixed**: Direct instantiation in `__main__` blocks
✅ **Safety Fixed**: No `sys.exit()` calls that could kill KiCad
✅ **Package Built**: Updated 96.4 KB package ready for installation

**The plugin should now appear in KiCad's External Plugins menu and execute properly as IPC plugins.**
