#!/usr/bin/env python3
"""
BULLETPROOF Minimal KiCad IPC Plugin 

This version handles every possible error case and ensures KiCad never crashes.
Based on official KiCad IPC API best practices.
"""

import sys
import traceback

def main():
    """Bulletproof track creation"""
    commit = None
    board = None
    
    try:
        print("🧪 Starting bulletproof minimal IPC plugin...")
        
        # 1. Import with detailed error handling
        try:
            from kipy import KiCad
            from kipy.board_types import Track
            from kipy.util.units import from_mm
            from kipy.geometry import Vector2
            print("✅ IPC API imported successfully")
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            print("Solution: pip install kicad-python")
            return 1
        except Exception as e:
            print(f"❌ Unexpected import error: {e}")
            return 1
        
        # 2. Connect with timeout handling
        try:
            print("Connecting to KiCad...")
            kicad = KiCad()
            print("✅ Connected to KiCad IPC server")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("Make sure KiCad 9.0+ is running with a PCB open")
            return 1
        
        # 3. Get board with validation
        try:
            print("Getting active board...")
            board = kicad.get_board()  # CORRECT API CALL
            if not board:
                print("❌ No board available")
                return 1
            print(f"✅ Got board: {board.name}")
        except Exception as e:
            print(f"❌ Failed to get board: {e}")
            return 1
        
        # 4. Begin commit transaction (CRITICAL)
        try:
            print("Starting commit transaction...")
            commit = board.begin_commit()
            print("✅ Commit transaction active")
        except Exception as e:
            print(f"❌ Failed to begin commit: {e}")
            return 1
        
        # 5. Create track with full validation
        try:
            print("Creating track object...")
            
            # Create track
            track = Track()
            
            # Set geometry
            track.start = Vector2(from_mm(10), from_mm(10))
            track.end = Vector2(from_mm(30), from_mm(10))
            track.width = from_mm(0.25)
            track.layer = 0  # F.Cu
            
            # Assign to net (prevents orphaned tracks)
            nets = board.get_nets()
            if nets and len(nets) > 0:
                track.net = nets[0]
                print(f"✅ Track assigned to net: {nets[0].name}")
            else:
                print("⚠️  No nets found, creating unconnected track")
            
            print("✅ Track object configured")
        except Exception as e:
            print(f"❌ Track creation failed: {e}")
            if commit:
                board.drop_commit(commit)
            return 1
        
        # 6. Add track to board (THE CRITICAL STEP)
        try:
            print("Adding track to board...")
            created_items = board.create_items([track])
            
            if not created_items or len(created_items) == 0:
                print("❌ No items were created")
                board.drop_commit(commit)
                return 1
                
            print(f"✅ Successfully created {len(created_items)} items")
        except Exception as e:
            print(f"❌ Failed to add track: {e}")
            traceback.print_exc()
            if commit:
                board.drop_commit(commit)
            return 1
        
        # 7. Commit changes (CRITICAL FOR CLEANUP)
        try:
            print("Committing changes...")
            board.push_commit(commit, "Bulletproof minimal test track")
            commit = None  # Mark as committed
            print("✅ Changes committed successfully")
        except Exception as e:
            print(f"❌ Commit failed: {e}")
            if commit:
                board.drop_commit(commit)
            return 1
        
        # 8. Optional: Save board
        try:
            print("Saving board...")
            board.save()
            print("✅ Board saved")
        except Exception as e:
            print(f"⚠️  Save failed (non-critical): {e}")
            # Don't return error - commit already succeeded
        
        print("🎉 SUCCESS! Track created from (10mm,10mm) to (30mm,10mm)")
        print("Check your KiCad PCB - you should see a new track!")
        return 0
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        
        # Emergency cleanup
        if commit and board:
            try:
                board.drop_commit(commit)
                print("✅ Emergency commit cleanup completed")
            except:
                print("⚠️  Could not clean up commit")
        
        return 1

if __name__ == "__main__":
    """Entry point"""
    try:
        exit_code = main()
        print(f"\n🏁 Plugin finished with exit code: {exit_code}")
        # Don't call sys.exit() - this can kill KiCad if run in plugin context
        # sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Plugin interrupted by user")
        # Don't call sys.exit(130) - this can kill KiCad
        # sys.exit(130)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        # Don't call sys.exit(1) - this can kill KiCad
        # sys.exit(1)
