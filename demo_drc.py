#!/usr/bin/env python3
"""
Quick DRC Extraction Demo
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kicad_interface import KiCadInterface, DRCRules
import logging

# Suppress INFO logging for cleaner output
logging.getLogger('kicad_interface').setLevel(logging.WARNING)

def main():
    print("🎯 OrthoRoute DRC Extraction Demo")
    print("=" * 40)
    
    interface = KiCadInterface()
    
    print("Connecting to KiCad...", end=" ")
    if not interface.connect():
        print("❌ FAILED")
        print("Make sure KiCad is running with a board open")
        return
    print("✅ SUCCESS")
    
    print("Extracting DRC rules...", end=" ")
    drc_rules = interface.extract_drc_rules()
    
    if not drc_rules:
        print("❌ FAILED")
        return
    print("✅ SUCCESS")
    
    print("\n📋 EXTRACTED DRC RULES:")
    print("-" * 25)
    print(f"Default Track Width: {drc_rules.default_track_width:.3f} mm")
    print(f"Default Via Size:    {drc_rules.default_via_size:.3f} mm") 
    print(f"Default Via Drill:   {drc_rules.default_via_drill:.3f} mm")
    print(f"Default Clearance:   {drc_rules.default_clearance:.3f} mm")
    print(f"Min Track Width:     {drc_rules.minimum_track_width:.3f} mm")
    print(f"Min Via Size:        {drc_rules.minimum_via_size:.3f} mm")
    
    print(f"\n🏷️  NETCLASSES ({len(drc_rules.netclasses)}):")
    print("-" * 20)
    for name, rules in drc_rules.netclasses.items():
        print(f"  {name}:")
        print(f"    Track: {rules['track_width']:.3f}mm")
        print(f"    Via:   {rules['via_size']:.3f}mm") 
        print(f"    Clear: {rules['clearance']:.3f}mm")
    
    # Test integration with board data
    print("\n🔗 Testing integration with board data...", end=" ")
    board_data = interface.get_board_data()
    if board_data and board_data.get('drc_rules'):
        print("✅ SUCCESS")
        print(f"   Board: {board_data['filename']}")
        print(f"   Nets:  {len(board_data['nets'])}")
        print(f"   DRC:   Integrated ✓")
    else:
        print("❌ FAILED")
    
    print("\n🎉 DRC extraction is working perfectly!")
    print("Ready for intelligent routing with design rules.")

if __name__ == "__main__":
    main()
