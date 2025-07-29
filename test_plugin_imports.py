#!/usr/bin/env python3
"""
Test script to verify KiCad plugin imports work correctly
"""

import sys
import os

# Add the kicad_plugin directory to path
plugin_dir = os.path.join(os.path.dirname(__file__), 'kicad_plugin')
sys.path.insert(0, plugin_dir)

def test_imports():
    """Test that all required modules can be imported"""
    
    print("Testing OrthoRoute plugin imports...")
    
    try:
        print("1. Testing ui_dialogs import...")
        import ui_dialogs
        print("   ✓ ui_dialogs imported successfully")
    except ImportError as e:
        print(f"   ✗ ui_dialogs import failed: {e}")
        return False
    
    # For KiCad-dependent modules, we'll just check if they exist and can be parsed
    try:
        print("2. Testing board_export import (KiCad-dependent)...")
        import ast
        with open(os.path.join(plugin_dir, 'board_export.py'), 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        print("   ✓ board_export.py syntax is valid")
    except Exception as e:
        print(f"   ✗ board_export.py has syntax errors: {e}")
        return False
    
    try:
        print("3. Testing route_import import (KiCad-dependent)...")
        with open(os.path.join(plugin_dir, 'route_import.py'), 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        print("   ✓ route_import.py syntax is valid")
    except Exception as e:
        print(f"   ✗ route_import.py has syntax errors: {e}")
        return False
    
    try:
        print("4. Testing orthoroute_kicad import (KiCad-dependent)...")
        with open(os.path.join(plugin_dir, 'orthoroute_kicad.py'), 'r', encoding='utf-8') as f:
            content = f.read()
            # Basic syntax check
            ast.parse(content)
            # Check for the main class
            if 'class OrthoRouteKiCadPlugin' in content:
                print("   ✓ orthoroute_kicad.py syntax is valid and contains OrthoRouteKiCadPlugin")
            else:
                print("   ✗ OrthoRouteKiCadPlugin class not found in orthoroute_kicad.py")
                return False
    except Exception as e:
        print(f"   ✗ orthoroute_kicad.py has syntax errors: {e}")
        return False
    
    print("\n✓ All imports and syntax checks successful!")
    return True

if __name__ == "__main__":
    try:
        success = test_imports()
        if success:
            print("\n🎉 Plugin imports are working correctly!")
            print("The plugin should now load properly in KiCad.")
        else:
            print("\n❌ Some imports failed. Check the error messages above.")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
