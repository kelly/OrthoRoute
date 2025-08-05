#!/usr/bin/env python3
"""
BARE MINIMUM WORKING KiCad IPC Plugin Example

This is the absolute simplest example that:
1. Connects to KiCad via IPC API
2. Creates one track
3. Commits it to the board 
4. Exits cleanly without crashing KiCad

Based on official KiCad IPC API documentation examples.
"""

import sys
import traceback
from pathlib import Path

def main():
    """Bare minimum working KiCad IPC plugin"""
    try:
        print("Starting bare minimum KiCad IPC plugin...")
        
        # Import the official kicad-python package
        try:
            from kipy import KiCad
            from kipy.board_types import Track
            from kipy.geometry import Vector2
            from kipy.util.units import from_mm
            print("✅ Successfully imported kicad-python (kipy)")
        except ImportError as e:
            print(f"❌ Failed to import kicad-python: {e}")
            print("Please install: pip install kicad-python")
            return 1
        
        # Connect to KiCad via IPC API
        print("Connecting to KiCad via IPC API...")
        try:
            kicad = KiCad()
            print("✅ Connected to KiCad IPC API")
        except Exception as e:
            print(f"❌ Failed to connect to KiCad: {e}")
            print("Make sure KiCad 9.0+ is running with IPC API enabled")
            return 1
        
        # Get the active board
        print("Getting active board...")
        try:
            board = kicad.board.get_board()
            print(f"✅ Got board: {board.name}")
        except Exception as e:
            print(f"❌ Failed to get board: {e}")
            print("Make sure a PCB is open in KiCad")
            return 1
            
        # Begin a commit transaction (critical for proper cleanup)
        print("Beginning commit transaction...")
        try:
            commit = board.begin_commit()
            print("✅ Commit transaction started")
        except Exception as e:
            print(f"❌ Failed to begin commit: {e}")
            return 1
        
        # Create a simple track
        print("Creating track...")
        try:
            # Create track object 
            track = Track()
            
            # Set start point (10mm, 10mm)
            track.start = Vector2(from_mm(10), from_mm(10))
            
            # Set end point (30mm, 10mm)  
            track.end = Vector2(from_mm(30), from_mm(10))
            
            # Set track width (0.25mm)
            track.width = from_mm(0.25)
            
            # Set layer to F.Cu (front copper)
            track.layer = 0  # F.Cu layer ID
            
            # Get the first net (or default net)
            nets = board.get_nets()
            if nets:
                track.net = nets[0]
                print(f"✅ Track assigned to net: {nets[0].name}")
            else:
                print("⚠️  No nets found, track will be unconnected")
            
            print("✅ Track object created")
        except Exception as e:
            print(f"❌ Failed to create track: {e}")
            print(f"Error details: {traceback.format_exc()}")
            # Drop the commit on error
            try:
                board.drop_commit(commit)
            except:
                pass
            return 1
        
        # Add track to board
        print("Adding track to board...")
        try:
            # This is the critical step - create_items adds the track
            created_items = board.create_items([track])
            print(f"✅ Track added to board: {len(created_items)} items created")
        except Exception as e:
            print(f"❌ Failed to add track to board: {e}")
            print(f"Error details: {traceback.format_exc()}")
            # Drop the commit on error
            try:
                board.drop_commit(commit)
            except:
                pass
            return 1
        
        # Commit the changes (critical step)
        print("Committing changes...")
        try:
            board.push_commit(commit, "Added test track via IPC API")
            print("✅ Changes committed successfully")
        except Exception as e:
            print(f"❌ Failed to commit changes: {e}")
            print(f"Error details: {traceback.format_exc()}")
            # Try to drop the commit
            try:
                board.drop_commit(commit)
            except:
                pass
            return 1
        
        # Refresh the board view (optional but recommended)
        print("Refreshing board view...")
        try:
            # Save the board to ensure changes are persisted
            board.save()
            print("✅ Board saved")
        except Exception as e:
            print(f"⚠️  Failed to save board: {e}")
            # This is not critical, continue
        
        print("✅ Plugin completed successfully!")
        print("You should now see a track from (10mm,10mm) to (30mm,10mm) on the board")
        return 0
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    """Entry point for plugin execution"""
    try:
        exit_code = main()
        print(f"🏁 Plugin finished with exit code: {exit_code}")
        # Don't call sys.exit() - this can kill KiCad if run in plugin context
        # sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        # Don't call sys.exit(1) - this can kill KiCad
        # sys.exit(1)
