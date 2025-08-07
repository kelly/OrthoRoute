# KiCad Plugin Toolbar Button Debugging Checklist

## What We Fixed
1. **Converted ultra_simple_test.py to proper ActionPlugin**: The file now inherits from `pcbnew.ActionPlugin` and calls `.register()`
2. **Fixed plugin.json runtime**: Changed from complex object to simple `"runtime": "ipc"`
3. **Created proper PCM package structure**: Package now includes `metadata.json` at root level + `plugins/` directory
4. **Confirmed file structure**: All required files (`icon.png`, plugin files) are present

## Correct PCM Package Structure
```
orthoroute-complete-pcm-package.zip
├── metadata.json          <- PCM package metadata (REQUIRED at root)
└── plugins/
    ├── plugin.json        <- Plugin configuration
    ├── ultra_simple_test.py <- ActionPlugin class
    ├── icon.png          <- Plugin icon
    └── [other plugin files]
```

## Step-by-Step Verification Process

### Step 1: Install the Complete PCM Package
✅ **Use the correct package**: `orthoroute-complete-pcm-package.zip`  
1. Open KiCad → Tools → Plugin and Content Manager
2. Click "Install from File"  
3. Select: `C:\Users\Benchoff\Documents\GitHub\OrthoRoute\orthoroute-complete-pcm-package.zip`
4. Wait for installation to complete

### Step 2: Check KiCad Plugin Detection
1. Open KiCad PCB Editor
2. Go to **Preferences → Preferences → PCB Editor → Action Plugins**
3. Look for "Ultra Simple Test" in the plugin list
4. If it appears:
   - ✅ Make sure "Show button" checkbox is **CHECKED**
   - ✅ Use arrow buttons to move it to desired position in toolbar
   - ✅ Click OK to save preferences

### Step 3: Verify Toolbar Button
1. Look at the top toolbar in PCB Editor
2. Should see a new button with your icon
3. If not visible, try:
   - Restart KiCad completely
   - Check if toolbar is too small (resize window)
   - Look for overflow arrows (`>>`) on toolbar

### Step 4: Check External Plugins Menu
1. Go to **Tools → External Plugins**
2. "Ultra Simple Test" should appear in this menu
3. Try clicking it - should show a message box

### Step 5: Check Log File
1. Look for file: `C:\Users\Benchoff\kicad_plugin_test.log`
2. Should contain messages about plugin loading and registration
3. If no log file exists, plugin isn't being loaded at all

## Troubleshooting Guide

### If Plugin Doesn't Appear in Preferences
- Check PCM installation location
- Verify all files are in plugins/ subdirectory
- Check log file for Python errors
- Try manual installation

### If Plugin Appears but No Toolbar Button
- Verify "Show button" is checked in preferences
- Restart KiCad after changing preferences  
- Check if toolbar has overflow (resize window)

### If Plugin Appears in Menu but Not Toolbar
- This is the "Show button" unchecked scenario
- Go to preferences and enable toolbar button

### If Plugin Shows Error When Run
- Check log file for detailed error messages
- Verify KiCad can import pcbnew and wx modules

## Expected Results
✅ Plugin appears in Action Plugins preferences  
✅ "Show button" can be checked/unchecked  
✅ Toolbar button appears when enabled  
✅ Clicking button shows "Success" message box  
✅ Plugin appears in Tools → External Plugins menu  
✅ Log file shows successful loading and registration  

## Next Steps if Still Not Working
1. Check KiCad version compatibility
2. Try creating minimal test plugin from scratch
3. Check KiCad Python console for import errors
4. Verify PCM vs manual installation differences
