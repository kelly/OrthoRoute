#!/usr/bin/env python3
"""
MINIMAL TRACK TEST PLUGIN
Just draws ONE track to test KiCad IPC API
"""

def main():
    """Draw one simple track"""
    print("🧪 MINIMAL TRACK TEST - Starting...")
    
    try:
        # Import KiCad API
        print("📥 Importing KiCad API...")
        from kipy import KiCad
        from kipy.board_types import Track
        from kipy.geometry import Vector2
        print("✅ API imported")
        
        # Connect to KiCad
        print("🔌 Connecting to KiCad...")
        kicad = KiCad()
        board = kicad.get_board()
        
        if not board:
            print("❌ No board found!")
            return
        print("✅ Connected to board")
        
        # Create ONE test track
        print("🔨 Creating test track...")
        
        track = Track()
        # Try different Vector2 constructor approaches
        start_vec = Vector2()
        start_vec.x = 10000000  # 10mm
        start_vec.y = 10000000  # 10mm
        track.start = start_vec
        
        end_vec = Vector2()
        end_vec.x = 30000000  # 30mm  
        end_vec.y = 10000000  # 10mm
        track.end = end_vec
        
        track.width = 250000  # 0.25mm wide
        track.layer = 0  # F.Cu
        
        print("   📍 Track: (10mm, 10mm) → (30mm, 10mm)")
        print("   📏 Width: 0.25mm")
        
        # Add to board
        print("📌 Adding to board...")
        board.create_items([track])
        
        # Save
        print("💾 Saving...")
        board.save()
        
        print("🎉 SUCCESS! Check for track at (10mm, 10mm) → (30mm, 10mm)")
        
    except ImportError as e:
        print(f"❌ API import failed: {e}")
        print("💡 Install: pip install kicad-python")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

# Entry point for KiCad External Plugin
def run():
    """Entry point for KiCad External Plugin system"""
    main()

if __name__ == "__main__":
    main()
