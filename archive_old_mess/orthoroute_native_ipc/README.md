# OrthoRoute Native IPC Plugin

This is the **Native IPC Plugin** version that creates an actual toolbar button in KiCad.

## Installation (Manual)

1. **Find your KiCad plugins directory:**
   - Windows: `C:\Users\<username>\Documents\KiCad\<version>\plugins\`
   - macOS: `~/Documents/KiCad/<version>/plugins/`
   - Linux: `~/.local/share/KiCad/<version>/plugins/`

2. **Create plugin directory:**
   ```
   mkdir orthoroute_gpu
   ```

3. **Copy all files to the plugin directory:**
   ```
   orthoroute_gpu/
   ├── plugin.json
   ├── orthoroute_gpu.py
   ├── icon.png
   └── README.md
   ```

4. **Restart KiCad completely**

5. **Open PCB Editor - look for the "OrthoRoute GPU" button in the toolbar!**

## How it Works

This uses KiCad's **Native IPC Plugin** system with the IPC schema (`https://go.kicad.org/api/schemas/v1`).

Key features:
- ✅ **Creates actual toolbar button** - `show-button: true`
- ✅ **Python runtime** - No executable needed
- ✅ **Icon support** - Light/dark theme icons
- ✅ **Direct integration** - Appears like built-in KiCad tools

## Difference from PCM Package

- **PCM Package**: Zip installation but no toolbar button (limitation of PCM)
- **Native IPC Plugin**: Manual installation but creates real toolbar button

This is the approach that will give you the button you want!
