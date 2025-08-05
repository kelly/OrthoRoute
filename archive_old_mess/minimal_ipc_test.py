#!/usr/bin/env python3
"""
Minimal IPC Plugin Test - For debugging KiCad plugin visibility
This is the simplest possible IPC plugin to test if the system works
"""

import sys
import os

def main():
    """Main entry point for minimal IPC test"""
    print("🧪 Minimal IPC Test Plugin Starting...")
    
    # Test 1: Basic Python execution
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} working")
    print(f"✅ Working directory: {os.getcwd()}")
    
    # Test 2: Try to import IPC API
    try:
        from kipy import KiCad
        print("✅ KiCad IPC API (kipy) imported successfully")
        ipc_available = True
    except ImportError as e:
        print(f"❌ Failed to import KiCad IPC API: {e}")
        print("   Install with: pip install kicad-python")
        ipc_available = False
    
    # Test 3: Try to connect to KiCad (if API available)
    if ipc_available:
        try:
            # Try to get environment variables
            api_socket = os.environ.get('KICAD_API_SOCKET')
            api_token = os.environ.get('KICAD_API_TOKEN')
            
            print(f"🔍 API Socket: {api_socket}")
            print(f"🔍 API Token: {'***' if api_token else 'None'}")
            
            if api_socket and api_token:
                kicad = KiCad()
                print("✅ Connected to KiCad via IPC API")
            else:
                print("⚠️  No IPC API environment variables (normal when run standalone)")
                
        except Exception as e:
            print(f"⚠️  IPC connection failed (normal if no active PCB): {e}")
    
    # Test 4: Success message
    print("🎯 Minimal IPC test completed!")
    print("If you see this message, the plugin executed successfully.")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        print(f"🏁 Test finished with exit code: {exit_code}")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
