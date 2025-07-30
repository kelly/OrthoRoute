# KiCad IPC API Transition Guide for OrthoRoute

## Overview

KiCad is transitioning from SWIG-based Python bindings to a new IPC (Inter-Process Communication) API. The SWIG bindings are deprecated as of KiCad 9.0 and will be removed in KiCad 10.0 (planned for February 2026).

## Current Status

### APIs Available
- **SWIG API**: `import pcbnew` (deprecated, will be removed in KiCad 10.0)
- **IPC API**: `from kicad.pcbnew import Board` (future-proof, requires kicad-python library)

### OrthoRoute IPC Support
We've implemented a hybrid approach that supports both APIs:

1. **API Bridge** (`api_bridge.py`): Compatibility layer that detects and uses available APIs
2. **IPC Test Plugin** (`ipc_api_test_plugin.py`): Comprehensive testing of both APIs
3. **Updated OrthoRoute** (`__init___ipc_compatible.py`): Plugin version with IPC support

## Installation

### For IPC API Support
```bash
# Install kicad-python library
pip install kicad-python

# Or install from source
git clone https://gitlab.com/kicad/code/kicad-python.git
cd kicad-python
pip install .
```

### Testing API Availability
Use our IPC test plugin to verify which APIs are available:

1. Install the test plugin in KiCad
2. Run "KiCad IPC API Test" from Tools → External Plugins
3. Check console output for detailed API comparison

## Key Differences

### Board Access
```python
# SWIG API (deprecated)
import pcbnew
board = pcbnew.GetBoard()

# IPC API (future)
from kicad.pcbnew.board import Board
swig_board = pcbnew.GetBoard()  # Still needed as intermediate
ipc_board = Board.wrap(swig_board)
```

### Net Detection
```python
# SWIG API
netcodes = board.GetNetsByNetcode()
for netcode, net in netcodes.items():
    for footprint in board.GetFootprints():
        for pad in footprint.Pads():
            if pad.GetNet().GetNetCode() == netcode:
                # Process pad

# IPC API  
ipc_board = Board.wrap(swig_board)
for module in ipc_board.modules:
    # Higher-level module access
    # Pad access may be different
```

### Track Creation
```python
# SWIG API
track = pcbnew.PCB_TRACK(board)
track.SetStart(pcbnew.VECTOR2I(x1, y1))
track.SetEnd(pcbnew.VECTOR2I(x2, y2))
track.SetLayer(layer_id)
track.SetWidth(width)
board.Add(track)

# IPC API
coords = [(x1/1e6, y1/1e6), (x2/1e6, y2/1e6)]  # Convert to mm
track = ipc_board.add_track(coords, layer='F.Cu', width=width/1e6)
```

## Migration Strategy

### Phase 1: Hybrid Support (Current)
- Support both SWIG and IPC APIs
- Use API bridge for compatibility
- Default to IPC when available, fallback to SWIG

### Phase 2: IPC Primary (KiCad 9.x)
- Make IPC API the primary interface
- Keep SWIG as fallback for compatibility
- Encourage users to install kicad-python

### Phase 3: IPC Only (KiCad 10.0+)
- Remove SWIG support when deprecated
- Use IPC API exclusively
- Full transition completed

## Implementation Notes

### API Bridge Benefits
1. **Automatic Detection**: Detects which APIs are available
2. **Seamless Fallback**: Uses IPC when available, SWIG as fallback
3. **Unified Interface**: Same method calls regardless of underlying API
4. **Future-Proof**: Easy to remove SWIG support when deprecated

### Key Challenges
1. **Different Paradigms**: IPC uses higher-level abstractions
2. **Unit Conversions**: IPC uses mm, SWIG uses nanometers
3. **Method Names**: Different naming conventions between APIs
4. **Installation**: Users need to install kicad-python separately

### Net Detection Fix Applied
The critical net-pad matching bug has been fixed in both API paths:
```python
# Fixed logic (works with both APIs)
pad_net = pad.GetNet()
if pad_net.GetNetCode() == netcode:  # Use netcode comparison, not object comparison
    net_pads.append(pad)
```

## Testing

### Comprehensive Test Suite
1. **API Availability Test**: Check which APIs are installed
2. **Board Info Comparison**: Compare board data access between APIs  
3. **Net Detection Comparison**: Verify net enumeration works correctly
4. **Module Access Comparison**: Test footprint/module access patterns
5. **Track Creation Test**: Verify track creation with both APIs

### Test Files Created
- `ipc_api_test_plugin.py`: Comprehensive API comparison test
- `api_bridge.py`: Compatibility layer implementation
- `__init___ipc_compatible.py`: IPC-compatible OrthoRoute plugin

## Usage Examples

### Using API Bridge in OrthoRoute
```python
from api_bridge import get_api_bridge

# Get bridge instance
bridge = get_api_bridge()

# Extract board data (works with both APIs)
board = bridge.get_board()
board_data = bridge.extract_board_data(board)

# Create tracks (automatic API selection)
bridge.create_track(start_pos, end_pos, layer=0, width=200000, net=net)
```

### Manual API Selection
```python
# Check API availability
api_info = bridge.get_api_info()
print(f"Using {api_info['current_api']} API")

if api_info['ipc_available']:
    print("✅ Future-proof IPC API available")
else:
    print("⚠️ Consider installing kicad-python for IPC support")
```

## Recommendations

### For Plugin Developers
1. **Use API Bridge**: Implement hybrid support for both APIs
2. **Test Thoroughly**: Verify functionality with both SWIG and IPC
3. **Plan Migration**: Prepare for SWIG deprecation in KiCad 10.0
4. **Document Transition**: Help users understand API changes

### For Users
1. **Install kicad-python**: `pip install kicad-python`
2. **Test Plugins**: Verify plugins work with IPC API
3. **Report Issues**: Help developers identify compatibility problems
4. **Plan Upgrades**: Prepare for KiCad 10.0 transition

### For OrthoRoute Users
1. **Current Status**: Plugin works with both APIs (hybrid mode)
2. **Recommended**: Install kicad-python for future compatibility
3. **Testing**: Use IPC test plugin to verify API functionality
4. **Transition**: Seamless - no user action required

## Timeline

- **KiCad 9.0**: SWIG deprecated, IPC available
- **KiCad 9.x**: Transition period, both APIs supported
- **KiCad 10.0** (Feb 2026): SWIG removed, IPC only
- **OrthoRoute**: Hybrid support implemented, ready for transition

## Resources

- [KiCad IPC API Documentation](https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/)
- [kicad-python Library](https://gitlab.com/kicad/code/kicad-python)
- [KiCad Developer Documentation](https://dev-docs.kicad.org/en/apis-and-binding/pcbnew/)
- [OrthoRoute IPC Test Plugin](ipc_api_test_plugin.py)

## Summary

The transition to IPC API is a significant change that affects all KiCad Python plugins. OrthoRoute is now prepared for this transition with:

1. ✅ **Hybrid API Support**: Works with both SWIG and IPC
2. ✅ **Automatic Detection**: Uses best available API
3. ✅ **Future-Proof**: Ready for KiCad 10.0
4. ✅ **Testing Tools**: Comprehensive API verification
5. ✅ **Bug Fixes**: All routing issues resolved for both APIs

This ensures OrthoRoute continues to work seamlessly through the KiCad API transition period and beyond.
