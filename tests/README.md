# OrthoRoute Tests

This directory contains test and debug files for the OrthoRoute project.

## Test Files

- `test_core.py` - Basic functionality tests (imports, dependencies)
- `test_pad_polygons.py` - Tests for polygon-based pad extraction
- `test_polygon_interface.py` - Tests for the KiCad interface polygon functions

## Debug Files

- `debug_bounds.py` - Debug board boundary extraction
- `debug_pads.py` - Debug pad data extraction
- `debug_pad_structure.py` - Debug pad structure analysis
- `debug_zones.py` - Debug copper zone extraction

## Running Tests

To run the basic test suite:

```bash
python tests/test_core.py
```

To run individual debug scripts:

```bash
python tests/debug_pads.py
python tests/test_pad_polygons.py
```

## Requirements

All tests require:
- Python 3.8+
- PyQt6
- kipy (KiCad IPC library)
- KiCad running with a board open for IPC tests
