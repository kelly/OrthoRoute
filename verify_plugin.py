"""
Simple plugin verification test
Run this from the KiCad plugin directory to verify plugin registration
"""
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("=== OrthoRoute Plugin Verification ===")
print(f"Plugin directory: {current_dir}")

try:
    # Test importing the plugin
    from orthoroute_kicad import OrthoRouteKiCadPlugin
    print("✓ Plugin import successful")
    
    # Create plugin instance
    plugin = OrthoRouteKiCadPlugin()
    print("✓ Plugin instance created")
    
    # Test plugin properties
    print(f"Plugin name: {plugin.GetName()}")
    print(f"Plugin description: {plugin.GetDescription()}")
    print(f"Plugin category: {plugin.GetCategoryName()}")
    
    # Test registration
    plugin.register()
    print("✓ Plugin registration successful")
    
    print("\n=== Plugin should be available in KiCad! ===")
    print("Look for 'OrthoRoute GPU Autorouter' in Tools menu")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
