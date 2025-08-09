#!/usr/bin/env python3
"""Debug bounds and coordinates"""

from kicad_interface import KiCadInterface

ki = KiCadInterface()
if ki.connect():
    data = ki.get_board_data()
    print(f'Board bounds: {data.get("bounds", "None")}')
    pads = data.get('pads', [])
    print(f'Number of pads: {len(pads)}')
    if pads:
        for i, pad in enumerate(pads[:5]):
            print(f'Pad {i}: x={pad.get("x", "?")} y={pad.get("y", "?")} size_x={pad.get("size_x", "?")} size_y={pad.get("size_y", "?")}')
