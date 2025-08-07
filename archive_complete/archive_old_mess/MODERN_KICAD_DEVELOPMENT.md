# KiCad IPC API Development Guide - OrthoRoute Implementation

## Executive Summary

This document outlines how OrthoRoute implements **modern KiCad plugin development** using the revolutionary IPC API architecture introduced in KiCad 9.0. Our implementation serves as a reference for the new standard of KiCad plugin development, demonstrating the transition from deprecated SWIG bindings to the stable, protocol-based IPC system.

## Understanding the Architectural Revolution

### SWIG → IPC API Transition

**KiCad 9.0 marks a fundamental shift** in plugin architecture:

- **SWIG Bindings (Deprecated)**: Direct access to KiCad's internal C++ APIs
  - ❌ Unstable across KiCad versions
  - ❌ Can crash KiCad process
  - ❌ Python-only
  - ❌ **Removed in KiCad 10.0**

- **IPC API (New Standard)**: Protocol-based inter-process communication
  - ✅ Stable Protocol Buffer interface
  - ✅ Process isolation prevents crashes
  - ✅ Language-agnostic
  - ✅ **Official KiCad roadmap through 10.0+**

### OrthoRoute's Implementation Strategy

OrthoRoute demonstrates **best practices for modern KiCad plugin development**:

1. **Pure IPC API**: No SWIG dependencies, future-proof architecture
2. **Process Isolation**: GPU operations in separate process for stability
3. **Professional Packaging**: Plugin and Content Manager integration
4. **Comprehensive Testing**: Minimal test plugin validates core functionality
5. **Modern Tooling**: CI/CD, automated packaging, and proper documentation

## Technical Implementation Details

### 1. IPC Connection Architecture

```python
# OrthoRoute IPC Connection Pattern
from kipy import KiCad
from kipy.board_types import Track
from kipy.util.units import from_mm
from kipy.geometry import Vector2

# Environment variables set by KiCad
socket_path = os.environ.get('KICAD_API_SOCKET')
token = os.environ.get('KICAD_API_TOKEN')

# Connect via Protocol Buffers over Unix socket
kicad = KiCad()
board = kicad.get_board()
```

### 2. Plugin Package Structure

```
orthoroute-plugin/
├── plugin.json          # IPC plugin definition (NOT ActionPlugin)
├── metadata.json        # Plugin and Content Manager metadata
├── orthoroute_main.py   # Main entry point with main() function
├── minimal_test.py      # API validation plugin
└── resources/
    └── icon.png         # Plugin icon
```

### 3. Plugin Definition (plugin.json)

```json
{
  "identifier": "com.github.bbenchoff.orthoroute",
  "name": "OrthoRoute GPU Autorouter",
  "requirements": {
    "packages": ["kicad-python>=0.4.0"]
  },
  "runtime": {
    "type": "python",
    "python_interpreter": "auto"
  },
  "actions": [{
    "identifier": "orthoroute.run",
    "entrypoint": "orthoroute_main.py"
  }]
}
```

### 4. Entry Point Pattern

```python
#!/usr/bin/env python3
def main():
    """IPC plugin entry point - must be main() function"""
    try:
        from kipy import KiCad
        # Plugin logic here
        return 0  # Success
    except Exception as e:
        print(f"Error: {e}")
        return 1  # Failure

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
```

## Key Differences from SWIG Plugins

| Aspect | SWIG (Deprecated) | IPC API (Modern) |
|--------|------------------|------------------|
| **Base Class** | `pcbnew.ActionPlugin` | Standalone script with `main()` |
| **Communication** | Direct C++ API calls | Protocol Buffers over Unix socket |
| **Process Model** | Runs in KiCad process | Separate process |
| **Crash Safety** | Can crash KiCad | Cannot crash KiCad |
| **Installation** | Manual file copying | Plugin and Content Manager |
| **Environment** | KiCad's Python | Managed virtual environment |
| **API Stability** | Breaks with KiCad updates | Stable across versions |

## Community Resources and Best Practices

### Official Documentation
- **IPC API Docs**: https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/
- **kicad-python Library**: https://docs.kicad.org/kicad-python-main/
- **Plugin Development**: https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/for-addon-developers/

### Community Examples
- **Modern Plugin Template**: https://github.com/adamws/kicad-plugin-template
- **MitjaNemec's Collections**: https://github.com/MitjaNemec/Kicad_action_plugins
- **Awesome KiCad**: https://github.com/joanbono/awesome-kicad

### Development Practices
1. **Start with minimal test plugin** (like our `minimal-track-test.zip`)
2. **Use modern tooling**: Hatch, CI/CD, automated testing
3. **Follow IPC patterns**: Process isolation, Protocol Buffers
4. **Professional packaging**: Plugin and Content Manager integration
5. **Comprehensive documentation**: API usage, installation, troubleshooting

## OrthoRoute as Reference Implementation

### What OrthoRoute Demonstrates

1. **Minimal Test Plugin** (`minimal-track-test.zip`):
   - Validates IPC API connection
   - Tests basic track creation
   - 50 lines of pure IPC code
   - Perfect starting point for developers

2. **Full Production Plugin** (`orthoroute-kicad-addon.zip`):
   - Complex GPU routing system
   - Process isolation architecture
   - Professional error handling
   - Comprehensive logging and debugging

3. **Modern Development Workflow**:
   - Automated package building
   - CI/CD integration ready
   - Proper documentation structure
   - Community contribution guidelines

### Learning Path for Developers

1. **Install and test** `minimal-track-test.zip`
2. **Study the source code** (50 lines, well-commented)
3. **Examine** full OrthoRoute implementation
4. **Use as template** for your own IPC plugins
5. **Contribute improvements** back to community

## Migration Strategy from SWIG

### For Existing SWIG Plugin Developers

1. **Assess Current Plugin**: Identify SWIG API usage patterns
2. **Install KiCad 9.0+**: Get IPC API support
3. **Test Minimal Plugin**: Validate IPC environment works
4. **Map SWIG→IPC APIs**: Use kicad-python documentation
5. **Rewrite Entry Point**: Convert ActionPlugin to main() function
6. **Update Package Structure**: plugin.json, metadata.json
7. **Test Thoroughly**: Use process isolation benefits
8. **Submit to PCM**: Use Plugin and Content Manager

### API Mapping Examples

```python
# SWIG (Old)
import pcbnew
board = pcbnew.GetBoard()
track = pcbnew.PCB_TRACK(board)

# IPC (New)  
from kipy import KiCad
from kipy.board_types import Track
board = kicad.get_board()
track = Track()
```

## Future-Proofing and Long-term Strategy

### KiCad 10.0 Preparation
- **SWIG removal confirmed** - all plugins must migrate
- **IPC API stability guaranteed** - Protocol Buffer versioning
- **Enhanced features planned** - More API coverage, better tooling

### Community Evolution
- **Professional development practices** becoming standard
- **Plugin and Content Manager** central distribution
- **Modern tooling adoption** (Hatch, CI/CD, testing)
- **Language diversification** beyond Python

## Conclusion

OrthoRoute represents the **gold standard for modern KiCad plugin development**, demonstrating how to embrace the IPC API architecture for stable, professional, and future-proof plugin development. Our minimal test plugin provides an ideal starting point for developers, while the full implementation showcases advanced patterns for complex functionality.

**The era of SWIG-based KiCad plugins is ending** - OrthoRoute shows the path forward with the IPC API, process isolation, and modern development practices that will define KiCad plugin development through KiCad 10.0 and beyond.

---

**Key Takeaway**: OrthoRoute is not just a routing plugin - it's a comprehensive reference implementation for the future of KiCad plugin development using the revolutionary IPC API architecture.
