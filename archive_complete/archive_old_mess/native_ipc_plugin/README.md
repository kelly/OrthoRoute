# OrthoRoute Native IPC Plugin - C++ Implementation

## Build Instructions

### Windows (Visual Studio)
```bash
cl /EHsc orthoroute_main.cpp /Fe:orthoroute.exe
```

### Windows (MinGW)
```bash
g++ -std=c++17 orthoroute_main.cpp -o orthoroute.exe
```

### Linux/macOS
```bash
g++ -std=c++17 orthoroute_main.cpp -o orthoroute
```

## Installation

1. Build the executable
2. Copy to KiCad plugins directory:
   - Windows: `C:\Users\<username>\Documents\KiCad\<version>\plugins\orthoroute\`
   - macOS: `/Users/<username>/Documents/KiCad/<version>/plugins/orthoroute/`
   - Linux: `~/.local/share/KiCad/<version>/plugins/orthoroute/`

3. Copy plugin.json to the same directory
4. Restart KiCad - toolbar buttons should appear!

## Next Steps

1. Implement Protocol Buffer communication with KiCad IPC API
2. Add routing algorithms
3. Create GUI for settings
4. Package for distribution

This approach gives you:
- Native C++ performance
- Direct toolbar integration
- Professional user experience
- No Python dependencies
