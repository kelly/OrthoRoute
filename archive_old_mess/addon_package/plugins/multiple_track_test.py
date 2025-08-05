#!/usr/bin/env python3
"""
Multiple Track Creation Methods Test
Try different ways to create tracks to see what actually works
"""

import sys

def main():
    """Test multiple methods of creating tracks"""
    print("🧪 MULTIPLE TRACK CREATION TEST")
    print("=" * 50)
    
    try:
        # Import API
        from kipy import KiCad
        from kipy.board import Board
        from kipy.board_types import Track
        from kipy.geometry import Vector2
        print("✅ API imported")
        
        # Connect
        kicad = KiCad()
        board = kicad.get_board()
        print("✅ Connected to board")
        
        # Method 1: Basic track, no net
        print("\n🧪 METHOD 1: Basic track without net")
        try:
            track1 = Track()
            track1.start = Vector2(5000000, 5000000)    # 5mm, 5mm
            track1.end = Vector2(25000000, 5000000)     # 25mm, 5mm
            track1.width = 200000  # 0.2mm
            track1.layer = 0
            
            board.create_items([track1])
            print("  ✅ Method 1 SUCCESS")
        except Exception as e:
            print(f"  ❌ Method 1 FAILED: {e}")
        
        # Method 2: Track with net
        print("\n🧪 METHOD 2: Track with net assignment")
        try:
            nets = board.get_nets()
            if nets:
                track2 = Track()
                track2.start = Vector2(5000000, 10000000)   # 5mm, 10mm
                track2.end = Vector2(25000000, 10000000)    # 25mm, 10mm
                track2.width = 200000
                track2.layer = 0
                track2.net = nets[0]
                
                board.create_items([track2])
                print(f"  ✅ Method 2 SUCCESS (net: {nets[0].name})")
            else:
                print("  ⚠️ Method 2 SKIPPED: No nets available")
        except Exception as e:
            print(f"  ❌ Method 2 FAILED: {e}")
        
        # Method 3: Multiple tracks at once
        print("\n🧪 METHOD 3: Multiple tracks")
        try:
            tracks = []
            for i in range(3):
                track = Track()
                track.start = Vector2(5000000, 15000000 + i * 2000000)
                track.end = Vector2(25000000, 15000000 + i * 2000000)
                track.width = 150000  # 0.15mm
                track.layer = 0
                tracks.append(track)
            
            board.create_items(tracks)
            print(f"  ✅ Method 3 SUCCESS: Created {len(tracks)} tracks")
        except Exception as e:
            print(f"  ❌ Method 3 FAILED: {e}")
        
        # Method 4: With commit transaction
        print("\n🧪 METHOD 4: Using commit transaction")
        try:
            commit = board.begin_commit()
            
            track4 = Track()
            track4.start = Vector2(5000000, 25000000)   # 5mm, 25mm
            track4.end = Vector2(25000000, 25000000)    # 25mm, 25mm
            track4.width = 300000  # 0.3mm thick
            track4.layer = 0
            
            board.create_items([track4])
            board.push_commit(commit, "Test track creation")
            print("  ✅ Method 4 SUCCESS: With transaction")
        except Exception as e:
            print(f"  ❌ Method 4 FAILED: {e}")
        
        # Force save
        print("\n💾 Saving board...")
        board.save()
        print("✅ Board saved")
        
        print("\n🎉 TRACK CREATION TEST COMPLETE!")
        print("Check your board for test tracks:")
        print("  • Method 1: y=5mm   (basic track)")
        print("  • Method 2: y=10mm  (with net)")
        print("  • Method 3: y=15,17,19mm (multiple)")
        print("  • Method 4: y=25mm  (with transaction)")
        
    except Exception as e:
        print(f"❌ Main error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
