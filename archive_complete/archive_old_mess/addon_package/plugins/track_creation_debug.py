#!/usr/bin/env python3
"""
OrthoRoute Track Creation Debug Tool
Tests multiple methods of creating tracks to find what actually works
"""

import sys
import traceback

# Import KiCad IPC API
try:
    from kipy import KiCad
    from kipy.board import Board
    from kipy.board_types import Track, Net, Via, FootprintInstance
    from kipy.util.units import to_mm, from_mm
    from kipy.geometry import Vector2
    print("✅ KiCad IPC API imported successfully")
    KIPY_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import KiCad IPC API: {e}")
    # Don't call sys.exit(1) - this kills KiCad entirely!
    KIPY_AVAILABLE = False

def main():
    """Test different track creation methods"""
    print("🧪 TRACK CREATION DEBUG TOOL")
    print("=" * 50)
    
    try:
        # Connect to KiCad
        kicad = KiCad()
        board = kicad.get_board()
        
        if not board:
            print("❌ No active board found!")
            return
        
        print("✅ Connected to board")
        
        # Method 1: Basic track creation
        test_basic_track_creation(board)
        
        # Method 2: Track with specific net
        test_track_with_net(board)
        
        # Method 3: Multiple tracks at once
        test_multiple_tracks(board)
        
        # Method 4: Track with different properties
        test_track_variations(board)
        
        print("\n🎯 All tests completed. Check your board for tracks!")
        
    except Exception as e:
        print(f"❌ Main error: {e}")
        traceback.print_exc()

def test_basic_track_creation(board):
    """Test 1: Basic track creation"""
    print("\n🧪 TEST 1: Basic track creation")
    
    try:
        track = Track()
        track.start = Vector2(5000000, 5000000)    # 5mm, 5mm
        track.end = Vector2(15000000, 5000000)     # 15mm, 5mm
        track.width = 250000  # 0.25mm
        track.layer = 0  # F.Cu
        
        print("  ✅ Track object created")
        
        # Try without net first
        board.create_items([track])
        board.save()
        
        print("  ✅ TEST 1 PASSED: Basic track created")
        
    except Exception as e:
        print(f"  ❌ TEST 1 FAILED: {e}")
        traceback.print_exc()

def test_track_with_net(board):
    """Test 2: Track with net assignment"""
    print("\n🧪 TEST 2: Track with net")
    
    try:
        # Get first available net
        nets = board.get_nets()
        if not nets:
            print("  ⚠️ No nets available, skipping")
            return
        
        net = nets[0]
        print(f"  Using net: {net.name}")
        
        track = Track()
        track.start = Vector2(5000000, 10000000)   # 5mm, 10mm
        track.end = Vector2(15000000, 10000000)    # 15mm, 10mm
        track.width = 200000  # 0.2mm
        track.layer = 0
        track.net = net
        
        print("  ✅ Track with net created")
        
        board.create_items([track])
        board.save()
        
        print("  ✅ TEST 2 PASSED: Track with net created")
        
    except Exception as e:
        print(f"  ❌ TEST 2 FAILED: {e}")
        traceback.print_exc()

def test_multiple_tracks(board):
    """Test 3: Multiple tracks at once"""
    print("\n🧪 TEST 3: Multiple tracks")
    
    try:
        tracks = []
        
        for i in range(3):
            track = Track()
            y_pos = 15000000 + (i * 2000000)  # 15mm + i*2mm
            
            track.start = Vector2(5000000, y_pos)
            track.end = Vector2(15000000, y_pos)
            track.width = 150000  # 0.15mm
            track.layer = 0
            
            tracks.append(track)
        
        print(f"  ✅ Created {len(tracks)} track objects")
        
        board.create_items(tracks)
        board.save()
        
        print("  ✅ TEST 3 PASSED: Multiple tracks created")
        
    except Exception as e:
        print(f"  ❌ TEST 3 FAILED: {e}")
        traceback.print_exc()

def test_track_variations(board):
    """Test 4: Different track properties"""
    print("\n🧪 TEST 4: Track variations")
    
    try:
        # Test different widths
        widths = [100000, 200000, 300000]  # 0.1mm, 0.2mm, 0.3mm
        
        for i, width in enumerate(widths):
            track = Track()
            y_pos = 25000000 + (i * 3000000)  # 25mm + i*3mm
            
            track.start = Vector2(5000000, y_pos)
            track.end = Vector2(15000000, y_pos) 
            track.width = width
            track.layer = 0
            
            board.create_items([track])
            print(f"  ✅ Track {i+1} (width {width/1000000:.1f}mm) created")
        
        board.save()
        print("  ✅ TEST 4 PASSED: Track variations created")
        
    except Exception as e:
        print(f"  ❌ TEST 4 FAILED: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
