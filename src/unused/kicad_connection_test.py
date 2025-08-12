#!/usr/bin/env python3
"""
KiCad IPC Connection Diagnostic Tool
Use this to debug connection issues with KiCad
"""

import os
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_kicad_connection():
    """Test KiCad IPC connection with detailed diagnostics"""
    
    print("=== KiCad IPC Connection Diagnostic ===\n")
    
    # Check environment
    print("1. Environment Check:")
    api_socket = os.environ.get('KICAD_API_SOCKET')
    api_token = os.environ.get('KICAD_API_TOKEN')
    print(f"   KICAD_API_SOCKET: {api_socket if api_socket else 'Not set'}")
    print(f"   KICAD_API_TOKEN: {'Set' if api_token else 'Not set'}")
    print()
    
    # Check kipy import
    print("2. Kipy Import Test:")
    try:
        import kipy
        from kipy import KiCad
        print(f"   ✓ kipy imported successfully from: {Path(kipy.__file__).parent}")
    except ImportError as e:
        print(f"   ✗ Failed to import kipy: {e}")
        return False
    print()
    
    # Test connection
    print("3. KiCad Connection Test:")
    try:
        print("   Creating KiCad client...")
        if api_socket or api_token:
            client = KiCad(socket_path=api_socket, kicad_token=api_token, timeout_ms=30000)
            print("   ✓ Using provided credentials")
        else:
            client = KiCad(timeout_ms=30000)
            print("   ✓ Using default connection")
        
        print("   Waiting for KiCad to be ready...")
        time.sleep(3)
        
        print("   Testing basic connection...")
        # Try a simple operation first
        try:
            docs = client.get_open_documents()
            print(f"   ✓ Connection successful - found {len(docs)} open documents")
        except Exception as e:
            print(f"   ⚠ get_open_documents failed: {e}")
            print("   Trying alternative connection method...")
        
        print("   Attempting to get board...")
        board = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                board = client.get_board()
                print(f"   ✓ Board retrieved successfully on attempt {attempt}")
                break
            except Exception as e:
                print(f"   ⚠ Attempt {attempt} failed: {e}")
                if "busy" in str(e).lower():
                    wait_time = 2 * attempt
                    print(f"   Waiting {wait_time}s for KiCad...")
                    time.sleep(wait_time)
                else:
                    break
        
        if board:
            print("   Testing board methods...")
            try:
                filename = getattr(board, 'name', 'Unknown')
                print(f"   ✓ Board name: {filename}")
                
                # Test a simple board operation
                nets = board.get_nets()
                print(f"   ✓ Found {len(nets)} nets")
                
                print("\n✅ All tests passed! KiCad connection is working.")
                return True
                
            except Exception as e:
                print(f"   ✗ Board operation failed: {e}")
                return False
        else:
            print("   ✗ Could not retrieve board")
            return False
            
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return False

def main():
    """Main diagnostic function"""
    print("KiCad IPC Diagnostic Tool")
    print("========================\n")
    
    print("Instructions:")
    print("1. Make sure KiCad is running")
    print("2. Have a PCB file open in KiCad")
    print("3. Enable the IPC API in KiCad (if not already enabled)")
    print("4. Run this script\n")
    
    success = test_kicad_connection()
    
    if success:
        print("\n🎉 KiCad connection is working properly!")
        print("You can now run the main OrthoRoute plugin.")
    else:
        print("\n❌ KiCad connection failed.")
        print("\nTroubleshooting steps:")
        print("1. Ensure KiCad is running and has a PCB file open")
        print("2. Check if KiCad IPC API is enabled")
        print("3. Try closing and reopening the PCB file in KiCad")
        print("4. Restart KiCad if necessary")
        print("5. Check KiCad version (9.0+ required for IPC API)")

if __name__ == "__main__":
    main()
