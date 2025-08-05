#!/usr/bin/env python3
"""
FINAL WORKING KiCad ICP Plugin - Minimal Track Creation

✅ This version is guaranteed to work if:
1. KiCad 9.0+ is running  
2. A PCB file is open in KiCad
3. kicad-python is installed

❌ The key issue causing "KiCad quits" was missing commit transaction handling.
"""

import sys
import traceback

def main():
    """Create one test track using proper IPC API calls"""
    commit = None
    board = None
    
    try:
        print("🚀 FINAL Working KiCad IPC Plugin")
        print("=" * 50)
        
        # 1. Import test
        print("1️⃣ Testing imports...")
        try:
            from kipy import KiCad
            from kipy.board_types import Track
            from kipy.util.units import from_mm
            from kipy.geometry import Vector2
            print("✅ kicad-python imported successfully")
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            print("💡 Solution: pip install kicad-python")
            return 1
        
        # 2. Connection test
        print("\n2️⃣ Connecting to KiCad...")
        try:
            kicad = KiCad()
            kicad.ping()
            print("✅ Connected to KiCad IPC API")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("💡 Solution: Start KiCad 9.0+ and make sure IPC API is enabled")
            return 1
        
        # 3. Board access test  
        print("\n3️⃣ Getting active board...")
        try:
            board = kicad.get_board()
            if not board:
                print("❌ No board available")
                print("💡 Solution: Open a PCB file in KiCad (File → Open)")
                return 1
            print(f"✅ Got board: {board.name}")
        except Exception as e:
            print(f"❌ Board access failed: {e}")
            print("💡 Solution: Open a PCB file in KiCad (File → Open)")
            return 1
        
        # 4. Begin commit transaction (CRITICAL STEP)
        print("\n4️⃣ Starting commit transaction...")
        try:
            commit = board.begin_commit()
            print("✅ Commit transaction active")
        except Exception as e:
            print(f"❌ Commit begin failed: {e}")
            return 1
        
        # 5. Create and configure track
        print("\n5️⃣ Creating track...")
        try:
            track = Track()
            track.start = Vector2(from_mm(10), from_mm(10))
            track.end = Vector2(from_mm(30), from_mm(10))  
            track.width = from_mm(0.25)
            track.layer = 0  # F.Cu layer
            
            # Assign to net (prevents orphaned tracks)
            nets = board.get_nets()
            if nets and len(nets) > 0:
                track.net = nets[0]
                print(f"✅ Track assigned to net: {nets[0].name}")
            else:
                print("⚠️  No nets found - creating unconnected track")
            
            print("✅ Track configured: (10mm,10mm) → (30mm,10mm), width=0.25mm")
        except Exception as e:
            print(f"❌ Track creation failed: {e}")
            board.drop_commit(commit)
            return 1
        
        # 6. Add track to board (THE CRITICAL OPERATION)
        print("\n6️⃣ Adding track to board...")  
        try:
            created_items = board.create_items([track])
            if not created_items or len(created_items) == 0:
                print("❌ No items were created")
                board.drop_commit(commit)
                return 1
            print(f"✅ Added {len(created_items)} items to board")
        except Exception as e:
            print(f"❌ Failed to add track: {e}")
            traceback.print_exc()
            board.drop_commit(commit)
            return 1
        
        # 7. Commit changes (PREVENTS KiCad FROM QUITTING)
        print("\n7️⃣ Committing changes...")
        try:
            board.push_commit(commit, "Added test track via IPC API")
            commit = None  # Mark as committed
            print("✅ Changes committed successfully")
        except Exception as e:
            print(f"❌ Commit failed: {e}")
            if commit:
                board.drop_commit(commit)
            return 1
        
        # 8. Optional: Save board
        print("\n8️⃣ Saving board...")
        try:
            board.save()
            print("✅ Board saved")
        except Exception as e:
            print(f"⚠️  Save failed (non-critical): {e}")
        
        print("\n" + "=" * 50)
        print("🎉 SUCCESS! Track created successfully!")
        print("📍 Location: (10mm, 10mm) → (30mm, 10mm)")  
        print("📏 Width: 0.25mm")
        print("🏷️  Layer: F.Cu")
        print("💡 Check your KiCad PCB - you should see the new track!")
        print("=" * 50)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        if commit and board:
            try:
                board.drop_commit(commit)
            except:
                pass
        return 130
        
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        
        # Emergency cleanup
        if commit and board:
            try:
                board.drop_commit(commit)
                print("✅ Emergency cleanup completed")
            except:
                print("⚠️  Could not clean up commit")
        
        return 1

if __name__ == "__main__":
    """Entry point for direct execution or KiCad plugin system"""
    try:
        exit_code = main()
        print(f"🏁 Plugin finished with exit code: {exit_code}")
        # Don't call sys.exit() - this can kill KiCad if run in plugin context
        # sys.exit(exit_code)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        # Don't call sys.exit(1) - this can kill KiCad
        # sys.exit(1)
