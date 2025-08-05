# KiCad IPC Plugin Visibility Troubleshooting Guide

## Current Status
✅ **Package Built Successfully**: `orthoroute-kicad-addon.zip` (91.3 KB)  
✅ **SWIG Registration Removed**: No legacy ActionPlugin registration  
✅ **IPC Plugin Structure**: Proper `plugin.json` with actions defined  
✅ **Entry Points Configured**: `orthoroute_ipc_plugin.py` as main entry point  

## Issue: Plugin Not Visible in KiCad UI

### Expected Behavior (KiCad 9.0+ IPC Plugins)
IPC plugins should appear in:
- **Tools → External Plugins** menu (as separate menu items)
- **Toolbar** (if `show_in_toolbar: true`)
- **Plugin and Content Manager** (for management)

### Possible Causes & Solutions

#### 1. **KiCad Version Compatibility**
**Check**: Ensure you're using KiCad 9.0+ with IPC API support
```bash
# In KiCad → Help → About KiCad
# Look for version 9.0 or higher
```

**Solution**: IPC plugins only work in KiCad 9.0+. Earlier versions need SWIG plugins.

#### 2. **Plugin Installation Location**
**Check**: Verify plugin is installed in correct directory
- Windows: `C:\Users\<username>\Documents\KiCad\9.0\plugins\orthoroute\`
- Linux: `~/.local/share/kicad/9.0/plugins/orthoroute/`
- macOS: `~/Documents/KiCad/9.0/plugins/orthoroute/`

**Debug Steps**:
1. After installing via PCM, check if the directory exists
2. Verify `plugin.json` is present in the plugin directory
3. Check file permissions (should be readable)

#### 3. **IPC API Service Status**
**Check**: Ensure KiCad's IPC API is enabled and running
```
KiCad → Preferences → Advanced Config → IPC API
Look for: api.enabled = true
```

**Debug**: Check if KiCad creates API socket files in temp directory

#### 4. **Plugin JSON Schema Issues**
**Current Schema**: Using PCM schema `https://go.kicad.org/pcm/schemas/v1`

**Potential Fix**: Try a simpler plugin.json structure:
```json
{
  "name": "OrthoRoute GPU Autorouter",
  "description": "GPU-accelerated autorouter",
  "version": "1.0.0",
  "actions": [
    {
      "identifier": "orthoroute.run",
      "name": "OrthoRoute GPU Autorouter",
      "entrypoint": "orthoroute_ipc_plugin.py"
    }
  ]
}
```

#### 5. **Python Environment Issues**
**Check**: Verify `kicad-python` is installed in KiCad's Python environment
```bash
# Windows (adjust path for your KiCad version)
"C:\Program Files\KiCad\9.0\bin\python.exe" -c "import kipy; print('IPC API available')"
```

**Solution**: Install in KiCad's Python environment:
```bash
"C:\Program Files\KiCad\9.0\bin\python.exe" -m pip install kicad-python
```

#### 6. **Plugin Entry Point Issues**
**Check**: Ensure entry point script is executable and has proper structure

**Current Structure**: ✅ Correct with `main()` function and proper imports

#### 7. **KiCad Console Output**
**Debug Steps**:
1. Open KiCad with console visible (Windows: shows automatically with debug builds)
2. Look for plugin loading messages
3. Check for IPC API connection errors
4. Monitor for Python import errors

## Alternative Testing Approach

### Create Minimal Test Plugin
Create a super-simple IPC plugin to test basic functionality:

**File**: `test_simple_ipc.py`
```python
#!/usr/bin/env python3
import sys
import os

def main():
    print("🧪 Simple IPC Test Plugin Executed!")
    
    # Test basic functionality
    try:
        from kipy import KiCad
        print("✅ KiCad IPC API imported successfully")
        
        # Try to connect (this might fail if no active PCB)
        try:
            kicad = KiCad()
            print("✅ Connected to KiCad via IPC API")
        except Exception as e:
            print(f"⚠️  IPC connection failed (normal if no active PCB): {e}")
            
    except ImportError as e:
        print(f"❌ Failed to import IPC API: {e}")
        return 1
    
    print("🎯 Test completed successfully!")
    return 0

if __name__ == "__main__":
    main()
```

**File**: `plugin.json` (minimal)
```json
{
  "name": "Simple IPC Test",
  "version": "1.0.0",
  "actions": [
    {
      "name": "Simple IPC Test",
      "entrypoint": "test_simple_ipc.py"
    }
  ]
}
```

## Debugging Commands

### 1. Check KiCad Plugin Directory
```powershell
# Windows
dir "C:\Users\$env:USERNAME\Documents\KiCad\9.0\plugins\"
```

### 2. Verify Plugin Installation
```powershell
# Check if OrthoRoute plugin directory exists
dir "C:\Users\$env:USERNAME\Documents\KiCad\9.0\plugins\orthoroute\"
```

### 3. Test Python Environment
```powershell
# Test KiCad's Python environment
& "C:\Program Files\KiCad\9.0\bin\python.exe" -c "import sys; print(sys.path)"
& "C:\Program Files\KiCad\9.0\bin\python.exe" -c "import kipy; print('IPC API OK')"
```

## Next Steps

1. **Verify KiCad Version**: Confirm you're using KiCad 9.0+
2. **Check Installation Location**: Verify plugin files are in correct directory  
3. **Test Minimal Plugin**: Try the simple test plugin above
4. **Check KiCad Console**: Look for error messages during plugin loading
5. **Try Manual Installation**: Extract zip contents directly to plugins directory

## Alternative: Manual Installation Test

1. Extract `orthoroute-kicad-addon.zip` 
2. Copy contents to: `C:\Users\<username>\Documents\KiCad\9.0\plugins\orthoroute\`
3. Restart KiCad completely
4. Check Tools → External Plugins

If manual installation works, the issue is with the PCM installation process.
If manual installation doesn't work, the issue is with the plugin structure or KiCad configuration.

## Support Information

When reporting this issue, please include:
- KiCad version (`Help → About KiCad`)
- Operating system version
- Plugin installation method (PCM vs manual)
- Contents of KiCad plugins directory
- Any console error messages
- Result of Python environment test commands above
