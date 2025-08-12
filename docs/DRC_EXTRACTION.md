# DRC (Design Rule Check) Information Extraction

This document explains how OrthoRoute extracts and uses Design Rule Check (DRC) information from KiCad boards for intelligent autorouting.

## Overview

OrthoRoute now extracts comprehensive DRC information from KiCad boards including:
- **NetClass definitions** with track widths, via sizes, and clearances
- **Per-net routing constraints** based on netclass assignments
- **Default routing parameters** for fallback routing
- **Minimum design rules** for constraint validation

## API Reference

### DRCRules Class

Contains all design rule information for a board:

```python
@dataclass
class DRCRules:
    netclasses: Dict[str, Dict]        # netclass_name -> rules dict
    default_track_width: float         # mm
    default_via_size: float           # mm  
    default_via_drill: float          # mm
    default_clearance: float          # mm
    minimum_track_width: float        # mm
    minimum_via_size: float           # mm
```

### NetClass Rules Format

Each netclass contains:

```python
{
    'name': 'Power',                  # NetClass name
    'track_width': 0.5,              # Track width in mm
    'via_size': 1.2,                 # Via diameter in mm
    'via_drill': 0.6,                # Via drill diameter in mm
    'clearance': 0.3                 # Minimum clearance in mm
}
```

## Usage Examples

### 1. Extract All DRC Rules

```python
from src.kicad_interface import KiCadInterface

interface = KiCadInterface()
interface.connect()

# Get board data with DRC rules
board_data = interface.get_board_data()
drc_rules = board_data['drc_rules']

print(f"Default track width: {drc_rules.default_track_width:.3f} mm")
print(f"NetClasses: {list(drc_rules.netclasses.keys())}")
```

### 2. Get Constraints for Specific Net

```python
# Get routing constraints for a specific net
net_name = "VCC"
constraints = interface.get_net_constraints(net_name)

print(f"Net '{net_name}' constraints:")
print(f"  NetClass: {constraints['netclass']}")
print(f"  Track width: {constraints['track_width']:.3f} mm")
print(f"  Via size: {constraints['via_size']:.3f} mm")
print(f"  Clearance: {constraints['clearance']:.3f} mm")
```

### 3. Use DRC Rules in Routing Algorithm

```python
def route_net_with_drc(net_name, board_data):
    drc_rules = board_data['drc_rules']
    
    # Get constraints for this specific net
    constraints = interface.get_net_constraints(net_name)
    
    # Use constraints in routing
    track_width = constraints['track_width']
    via_size = constraints['via_size'] 
    clearance = constraints['clearance']
    
    # Route with appropriate parameters
    route_tracks(
        track_width=track_width,
        via_size=via_size,
        clearance=clearance
    )
```

## KiCad API Methods Used

The DRC extraction uses these KiCad Python API methods:

### Project Level
```python
project = board.get_project()
netclasses = project.get_net_classes()
```

### Board Level  
```python
nets = board.get_nets()
netclass_mapping = board.get_netclass_for_nets(nets)
```

### NetClass Properties
```python
netclass.name           # NetClass name
netclass.track_width    # Track width (nanometers)
netclass.via_size       # Via diameter (nanometers)  
netclass.via_drill      # Via drill diameter (nanometers)
netclass.clearance      # Minimum clearance (nanometers)
```

## Error Handling

The DRC extraction includes robust error handling:

- **Fallback DRC Rules**: If extraction fails, provides sensible defaults
- **Unit Conversion**: Automatically converts from KiCad nanometers to millimeters
- **Property Variations**: Handles different KiCad API versions with multiple property names
- **Partial Extraction**: Continues even if some netclasses fail to extract

## Integration with Routing Algorithms

DRC rules can be integrated with routing algorithms for:

### 1. Track Width Selection
```python
def get_track_width(net_name, drc_rules):
    constraints = get_net_constraints(net_name)
    return max(constraints['track_width'], drc_rules.minimum_track_width)
```

### 2. Via Sizing
```python
def get_via_parameters(net_name, drc_rules):
    constraints = get_net_constraints(net_name)
    return {
        'diameter': max(constraints['via_size'], drc_rules.minimum_via_size),
        'drill': constraints['via_drill']
    }
```

### 3. Clearance Checking
```python
def check_clearance(track1, track2, drc_rules):
    clearance_required = max(
        track1.constraints['clearance'],
        track2.constraints['clearance']
    )
    return distance_between(track1, track2) >= clearance_required
```

## Testing

Use the provided test script to verify DRC extraction:

```bash
python test_drc_extraction.py
```

This will:
1. Connect to KiCad
2. Extract board DRC rules
3. Display all netclasses and their parameters
4. Test net-specific constraint retrieval

## Future Enhancements

Planned improvements for DRC integration:

1. **Differential Pair Rules**: Extract differential pair spacing and coupling
2. **Layer-Specific Rules**: Handle different rules per copper layer
3. **Advanced Constraints**: Support for length matching and impedance control
4. **Design Rule Validation**: Real-time checking during routing
5. **Custom Rule Support**: User-defined routing constraints

## Troubleshooting

### Common Issues

**NetClass not found:**
- Ensure the KiCad project file (.kicad_pro) is saved
- Check that netclasses are properly defined in KiCad

**Zero or invalid values:**
- KiCad API may return values in different units
- Check KiCad version compatibility
- Verify netclass definitions in KiCad

**Connection errors:**
- Ensure KiCad is running with a board open
- Check that KiCad API is enabled
- Verify environment variables (KICAD_API_SOCKET, KICAD_API_TOKEN)

### Debug Logging

Enable debug logging to see detailed DRC extraction:

```python
import logging
logging.getLogger('kicad_interface').setLevel(logging.DEBUG)
```

This will show:
- NetClass discovery process
- Property extraction attempts
- Unit conversion details
- Fallback rule activation
