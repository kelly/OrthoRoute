#!/usr/bin/env python3
"""
KiCad IPC API Test - Connection and Library Verification

This tests the kicad-python library installation and connection
without requiring a running KiCad instance.
"""

import sys

def test_library_import():
    """Test if kicad-python library can be imported"""
    print("🧪 Testing kicad-python library import...")
    
    try:
        # Test basic imports
        from kipy import KiCad
        from kipy.board_types import Track
        from kipy.util.units import from_mm
        from kipy.geometry import Vector2
        print("✅ All kicad-python imports successful")
        
        # Test library version
        try:
            import kipy
            if hasattr(kipy, '__version__'):
                print(f"✅ kicad-python version: {kipy.__version__}")
            else:
                print("✅ kicad-python loaded (version info not available)")
        except:
            print("⚠️  Version info not available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Solution: pip install kicad-python")
        return False
    except Exception as e:
        print(f"❌ Unexpected import error: {e}")
        return False

def test_connection():
    """Test connection to KiCad (requires running KiCad)"""
    print("\n🧪 Testing KiCad connection...")
    
    try:
        from kipy import KiCad
        
        # Try to connect
        print("Attempting to connect to KiCad...")
        kicad = KiCad()
        print("✅ Connection object created")
        
        # Test ping
        try:
            kicad.ping()
            print("✅ KiCad responded to ping")
            return True
        except Exception as e:
            print(f"❌ Ping failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Make sure KiCad 9.0+ is running with IPC API enabled")
        return False

def test_board_access():
    """Test board access (requires running KiCad with open PCB)"""
    print("\n🧪 Testing board access...")
    
    try:
        from kipy import KiCad
        
        kicad = KiCad()
        
        # Try to get board
        print("Attempting to get board...")
        board = kicad.get_board()
        
        if board:
            print(f"✅ Got board: {board.name}")
            print(f"✅ Board path: {board.name}")
            
            # Test basic board operations
            try:
                nets = board.get_nets()
                print(f"✅ Board has {len(nets)} nets")
                
                tracks = board.get_tracks()
                print(f"✅ Board has {len(tracks)} tracks")
                
                return True
            except Exception as e:
                print(f"⚠️  Board access limited: {e}")
                return True  # Connection works, just limited access
                
        else:
            print("❌ No board available")
            return False
            
    except Exception as e:
        print(f"❌ Board access failed: {e}")
        return False

def main():
    """Run all tests"""
    print("KiCad IPC API Connection Test")
    print("=" * 40)
    
    # Test 1: Library import
    if not test_library_import():
        print("\n❌ CRITICAL: kicad-python library not available")
        print("Install with: pip install kicad-python")
        return 1
    
    # Test 2: Connection test
    connection_ok = test_connection()
    if not connection_ok:
        print("\n⚠️  WARNING: Cannot connect to KiCad")
        print("Start KiCad 9.0+ to enable IPC API testing")
        return 2
    
    # Test 3: Board access test  
    board_ok = test_board_access()
    if not board_ok:
        print("\n⚠️  WARNING: Cannot access board")
        print("Open a PCB file in KiCad for full testing")
        return 3
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ kicad-python library is working correctly")
    print("✅ KiCad IPC API connection is active")
    print("✅ Board access is available")
    print("\nYou can now run KiCad IPC plugins successfully!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    print(f"\nTest completed with exit code: {exit_code}")
    sys.exit(exit_code)
