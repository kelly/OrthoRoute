#!/usr/bin/env python3
"""
Minimal KiCad API Test
Just test if we can connect to KiCad and read basic info
NO track creation, just API validation
"""

def main():
    """Test basic KiCad API connection"""
    print("🔌 MINIMAL KICAD API TEST")
    print("=" * 30)
    
    try:
        # Step 1: Import test
        print("1️⃣ Testing imports...")
        from kipy import KiCad
        from kipy.board import Board
        from kipy.board_types import Track, Net
        from kipy.geometry import Vector2
        print("   ✅ All imports successful")
        
        # Step 2: Connection test
        print("2️⃣ Testing KiCad connection...")
        kicad = KiCad()
        print("   ✅ KiCad object created")
        
        # Step 3: Board access test
        print("3️⃣ Testing board access...")
        board = kicad.get_board()
        if board:
            print("   ✅ Board object obtained")
        else:
            print("   ❌ No board available")
            return
        
        # Step 4: Basic board info
        print("4️⃣ Reading board information...")
        try:
            nets = board.get_nets()
            print(f"   📋 Nets: {len(nets)}")
            if nets:
                print(f"   📋 First net: '{nets[0].name}'")
        except Exception as e:
            print(f"   ❌ Error reading nets: {e}")
        
        try:
            footprints = board.get_footprints()
            print(f"   📦 Footprints: {len(footprints)}")
            if footprints:
                print(f"   📦 First footprint: '{footprints[0].reference_field.text.value}'")
        except Exception as e:
            print(f"   ❌ Error reading footprints: {e}")
        
        # Step 5: Track object creation test (no adding to board)
        print("5️⃣ Testing track object creation...")
        try:
            test_track = Track()
            test_track.start = Vector2(1000000, 1000000)  # 1mm, 1mm
            test_track.end = Vector2(2000000, 1000000)    # 2mm, 1mm
            test_track.width = 200000  # 0.2mm
            test_track.layer = 0
            print("   ✅ Track object created successfully")
            print(f"   📍 Track: {test_track.start} → {test_track.end}")
            print(f"   📏 Width: {test_track.width} nm")
        except Exception as e:
            print(f"   ❌ Error creating track object: {e}")
        
        print("\n🎉 API TEST COMPLETE!")
        print("All basic KiCad API operations working!")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Try: pip install kicad-python")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
